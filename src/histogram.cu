#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering (for input tiling),
// vectorized (int4) loads, and per-warp histograms.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Dynamic shared memory is split into two regions:
    // (1) Two input tile buffers (tile0 and tile1).
    // (2) Per-warp histograms (each of size numBins).
    extern __shared__ int sharedMem[];
    
    // Each block has blockDim.x * blockDim.y threads.
    int blockThreads = blockDim.x * blockDim.y;
    // For vectorized loading, each thread loads 4 ints at once.
    // Thus, one tile's size (in ints) is: tileSizeInts = blockThreads * 4.
    int tileSizeInts = blockThreads * 4;
    // Set up two buffers for double buffering:
    int *tile0 = sharedMem;                          // first tile buffer (tileSizeInts ints)
    int *tile1 = sharedMem + tileSizeInts;             // second tile buffer (tileSizeInts ints)
    // The rest of shared memory is used for per-warp histograms.
    const int warpSize = 32;
    int numWarps = blockThreads / warpSize; // assume blockThreads is a multiple of 32
    int *warpHist = (int*)(sharedMem + 2 * tileSizeInts); // region for per-warp histograms

    // Flatten the 2D thread index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Initialize per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Compute the global stride for tiles (in ints).
    int globalTileSizeInts = gridDim.x * tileSizeInts;

    // We'll use double buffering to load tiles from global memory.
    // Determine the first offset for this block.
    int firstOffset = blockIdx.x * tileSizeInts;
    // Load the first tile into tile0.
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;
        if (globalIndex + 3 < N) {
            int4 tmp = ((const int4*)data)[globalIndex / 4];
            tile0[tid*4 + 0] = tmp.x;
            tile0[tid*4 + 1] = tmp.y;
            tile0[tid*4 + 2] = tmp.z;
            tile0[tid*4 + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile0[tid*4 + i] = (idx < N) ? data[idx] : -1;
            }
        }
    }
    __syncthreads();

    // Process tiles in a double-buffered pipelined loop.
    // Start at the second tile.
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        // Load next tile into tile1.
        int globalIndex = offset + tid * 4;
        if (globalIndex + 3 < N) {
            int4 tmp = ((const int4*)data)[globalIndex / 4];
            tile1[tid*4 + 0] = tmp.x;
            tile1[tid*4 + 1] = tmp.y;
            tile1[tid*4 + 2] = tmp.z;
            tile1[tid*4 + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile1[tid*4 + i] = (idx < N) ? data[idx] : -1;
            }
        }
        __syncthreads();

        // Process the tile in tile0.
        {
            int localBin = -1;
            int localCount = 0;
            for (int i = tid; i < tileSizeInts; i += blockThreads) {
                int value = tile0[i];
                if (value < 0) continue;
                int bin = value / (1024 / numBins);
                if (bin == localBin) {
                    localCount++;
                } else {
                    if (localCount > 0) {
                        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                    }
                    localBin = bin;
                    localCount = 1;
                }
            }
            if (localCount > 0) {
                atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
            }
        }
        __syncthreads();

        // Swap buffers: tile1 becomes the current tile for next iteration.
        int *temp = tile0;
        tile0 = tile1;
        tile1 = temp;
        __syncthreads();
    }

    // Process the final tile loaded in tile0.
    {
        int localBin = -1;
        int localCount = 0;
        for (int i = tid; i < tileSizeInts; i += blockThreads) {
            int value = tile0[i];
            if (value < 0) continue;
            int bin = value / (1024 / numBins);
            if (bin == localBin) {
                localCount++;
            } else {
                if (localCount > 0) {
                    atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                }
                localBin = bin;
                localCount = 1;
            }
        }
        if (localCount > 0) {
            atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a single block-level (partial) histogram.
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            partialHist[blockIdx.x * numBins + i] = sum;
        }
    }
}

// Reduction kernel: Sum partial histograms from all blocks into the final histogram.
__global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < numBins) {
        int sum = 0;
        for (int b = 0; b < numBlocks; b++) {
            sum += partialHist[b * numBins + bin];
        }
        finalHist[bin] = sum;
    }
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
