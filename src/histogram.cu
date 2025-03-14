#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering (for input tiling),
// vectorized (int4) loads, and per-warp histograms.
// Configuration constants for the optimized histogram kernel
#define WARP_SIZE 32
#define HISTO_BINS_PER_THREAD 4  // Each thread handles multiple bins to reduce atomic ops
#define ITEMS_PER_THREAD 8       // Increased data processing per thread

// Align data to avoid bank conflicts
#define ALIGN_UP(x, size) (((x) + (size) - 1) & (~((size) - 1)))

__global__ void histogram_optimized_kernel(const int *__restrict__ data, int *__restrict__ partialHist, 
                                          int N, int numBins) {
    // Compute block and grid dimensions
    const int blockThreads = blockDim.x * blockDim.y;
    const int numWarps = blockThreads / WARP_SIZE;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Allocate shared memory with padding to avoid bank conflicts
    extern __shared__ int sharedMem[];
    
    // Calculate sizes with alignment to prevent bank conflicts
    const int tileSizeInts = blockThreads * ITEMS_PER_THREAD;
    const int alignedTileSize = ALIGN_UP(tileSizeInts, 32); // Align to 32 for bank conflict avoidance
    
    // Two tile buffers for double buffering with alignment
    int *tile0 = sharedMem;
    int *tile1 = sharedMem + alignedTileSize;
    
    // Per-warp histograms with padding to avoid bank conflicts
    const int paddedBinSize = ALIGN_UP(numBins, 32);
    int *warpHist = (int*)(sharedMem + 2 * alignedTileSize);
    
    // Thread-local registers for histogram updates (reduces atomic operations)
    int localBins[HISTO_BINS_PER_THREAD];
    int localCounts[HISTO_BINS_PER_THREAD];
    
    // Initialize thread-local histograms
    #pragma unroll
    for (int i = 0; i < HISTO_BINS_PER_THREAD; i++) {
        localBins[i] = -1;
        localCounts[i] = 0;
    }
    
    // Initialize warp histograms in shared memory
    // Use warp-level parallelism for faster initialization
    for (int i = lane; i < numWarps * paddedBinSize; i += WARP_SIZE) {
        if (i % paddedBinSize < numBins) {
            warpHist[i] = 0;
        }
    }
    __syncthreads();
    
    // Persistent thread approach: each thread processes multiple chunks
    const int globalTileSizeInts = gridDim.x * tileSizeInts;
    const unsigned int binStep = 1024 / numBins; // Precompute this division
    
    // Process input using persistent threads pattern
    for (int baseOffset = blockIdx.x * tileSizeInts; baseOffset < N; baseOffset += globalTileSizeInts) {
        // Calculate this thread's global indices
        int globalIdx = baseOffset + tid * ITEMS_PER_THREAD;
        
        // Register-based buffering for loaded values
        int values[ITEMS_PER_THREAD];
        
        // Vectorized loading with boundary checking
        if (globalIdx + ITEMS_PER_THREAD <= N) {
            // Fully aligned load when possible
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD/4; i++) {
                int loadIdx = globalIdx + i*4;
                int4 tmp = ((const int4*)data)[loadIdx/4];
                values[i*4] = tmp.x;
                values[i*4+1] = tmp.y;
                values[i*4+2] = tmp.z;
                values[i*4+3] = tmp.w;
            }
        } else {
            // Handle boundary case with scalar loads
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                int idx = globalIdx + i;
                values[i] = (idx < N) ? data[idx] : -1;
            }
        }
        
        // Process values using register-based buffering to reduce atomic operations
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int value = values[i];
            if (value >= 0) {
                int bin = value / binStep;
                
                // Try to accumulate in local registers first
                bool foundSlot = false;
                #pragma unroll
                for (int j = 0; j < HISTO_BINS_PER_THREAD; j++) {
                    if (bin == localBins[j]) {
                        localCounts[j]++;
                        foundSlot = true;
                        break;
                    } else if (localBins[j] == -1) {
                        localBins[j] = bin;
                        localCounts[j] = 1;
                        foundSlot = true;
                        break;
                    }
                }
                
                // If no available slot, flush the least frequently used one
                if (!foundSlot) {
                    // Find min count position
                    int minPos = 0;
                    int minCount = localCounts[0];
                    
                    #pragma unroll
                    for (int j = 1; j < HISTO_BINS_PER_THREAD; j++) {
                        if (localCounts[j] < minCount) {
                            minCount = localCounts[j];
                            minPos = j;
                        }
                    }
                    
                    // Flush the least used bin to shared memory
                    int binToFlush = localBins[minPos];
                    int countToFlush = localCounts[minPos];
                    atomicAdd(&warpHist[warp_id * paddedBinSize + binToFlush], countToFlush);
                    
                    // Store new value
                    localBins[minPos] = bin;
                    localCounts[minPos] = 1;
                }
            }
        }
        
        // Use warp-level synchronization instead of block-level when possible
        __syncwarp();
    }
    
    // Flush all remaining local counts to warp histograms
    #pragma unroll
    for (int i = 0; i < HISTO_BINS_PER_THREAD; i++) {
        if (localBins[i] >= 0 && localCounts[i] > 0) {
            atomicAdd(&warpHist[warp_id * paddedBinSize + localBins[i]], localCounts[i]);
        }
    }
    
    // Make sure all warps have completed their histograms
    __syncthreads();
    
    // Reduce per-warp histograms into block-level histogram using collaborative approach
    if (warp_id == 0) {
        // Each thread in the first warp handles multiple bins
        for (int binBase = lane; binBase < numBins; binBase += WARP_SIZE) {
            int sum = 0;
            
            // Accumulate across all warps
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * paddedBinSize + binBase];
            }
            
            // Write to global memory
            partialHist[blockIdx.x * numBins + binBase] = sum;
        }
    }
}

// Optimized reduction kernel using warp-level primitives and shuffle instructions
__global__ void histogram_reduce_kernel(const int *__restrict__ partialHist, 
                                       int *__restrict__ finalHist, 
                                       int numBins, int numBlocks) {
    // Thread and block identifiers
    const int tid = threadIdx.x;
    const int binIdx = blockIdx.x * blockDim.x + tid;
    const int warpId = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Shared memory for warp-level reductions
    extern __shared__ int sharedData[];
    
    // Process bins in grid-stride loop for better occupancy
    for (int bin = binIdx; bin < numBins; bin += gridDim.x * blockDim.x) {
        // Register for local accumulation
        int sum = 0;
        
        // Grid-stride loop to process multiple blocks per thread
        #pragma unroll 4
        for (int b = 0; b < numBlocks; b += 4) {
            // Coalesced loads from global memory
            if (b < numBlocks) sum += partialHist[b * numBins + bin];
            if (b+1 < numBlocks) sum += partialHist[(b+1) * numBins + bin];
            if (b+2 < numBlocks) sum += partialHist[(b+2) * numBins + bin];
            if (b+3 < numBlocks) sum += partialHist[(b+3) * numBins + bin];
        }
        
        // If this thread processes a valid bin, write the result
        if (bin < numBins) {
            finalHist[bin] = sum;
        }
    }
}

// Function to calculate the required shared memory size
inline size_t getHistogramSharedMemorySize(int blockThreads, int numBins) {
    const int tileSizeInts = blockThreads * ITEMS_PER_THREAD;
    const int alignedTileSize = ALIGN_UP(tileSizeInts, 32);
    const int numWarps = blockThreads / WARP_SIZE;
    const int paddedBinSize = ALIGN_UP(numBins, 32);
    
    // Two tile buffers + per-warp histograms with padding
    return (2 * alignedTileSize + numWarps * paddedBinSize) * sizeof(int);
}

int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [BlockSize] [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize] [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // Optionally accept block and grid sizes.
    int blockSizeTotal = 256; // default total threads per block
    int gridSize;
    if (argc >= 5)
        blockSizeTotal = atoi(argv[4]);
    if (argc >= 6)
        gridSize = atoi(argv[5]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    // Set 2D block shape: force blockDim.x = 32, blockDim.y = blockSizeTotal / 32.
    int blockDimX = 32;
    int blockDimY = blockSizeTotal / 32;
    if (blockDimY < 1) blockDimY = 1;
    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridSize);
    
    // Calculate shared memory size:
    //   Two tile buffers: 2 * tileSizeInts, where tileSizeInts = block.x * block.y * 4.
    //   Plus per-warp histogram: numWarps * numBins integers.
    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    
    // Allocate and initialize host memory.
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;  // values in [0, 1023]
    }
    
    // Allocate device memory.
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    
    // Copy input data to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    
    // Create CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    // Launch the optimized histogram kernel.
    histogram_optimized_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
    // Launch the reduction kernel.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate measured throughput based on approximate atomic operations.
    double totalOps = (double) N + (gridSize * numBins); // approximate total operations
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // Calculate occupancy.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;  // e.g., 2048 for V100
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSizeTotal, sharedMemSize);
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;  // percentage
    printf("Occupancy per SM: %f %%\n", occupancy);
    
    // Acquire GPU specifications and calculate theoretical peak integer operations per second.
    // For a Volta-like GPU, assume 64 integer (or FP32) cores per SM and 2 ops per cycle.
    int coresPerSM = 64;
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    // deviceProp.clockRate is in kHz; convert to Hz:
    double clockHz = deviceProp.clockRate * 1000.0;
    double theoreticalOps = totalCores * clockHz * 2;  // 2 ops per cycle
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", totalCores);
    printf("Clock Rate: %0.2f GHz\n", clockHz / 1e9);
    printf("Theoretical Peak Ops/sec (int): %e ops/sec\n", theoreticalOps);
    
    // (Optional) Copy final histogram from device to host and print nonzero bins.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Bin %d: %d\n", i, h_finalHist[i]);
    }
    
    // Clean up.
    free(h_data);
    free(h_finalHist);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
