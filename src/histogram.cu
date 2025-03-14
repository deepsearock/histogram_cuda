#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering, vectorized loads,
// per-warp histograms, bit-shift based bin calculation, and loop unrolling.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory layout:
    // [tile0 | tile1 | per-warp histograms]
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    const int blockThreads = blockDim.x * blockDim.y;
    // Each thread loads 32 integers.
    const int intsPerThread = 32;
    const int tileSizeInts = blockThreads * intsPerThread; // Total integers in a tile

    // Pointers for double-buffered tile buffers.
    int *tile0 = sharedMem;                    
    int *tile1 = sharedMem + tileSizeInts;       

    // Per-warp histogram region follows the two tile buffers.
    const int numWarps = blockThreads / warpSize;
    int *warpHist = sharedMem + 2 * tileSizeInts;  // length = numWarps * numBins

    // Precompute the bit-shift factor.
    int k = 0;
    int temp = numBins;
    while (temp > 1) {
        k++;
        temp >>= 1;
    }
    const int shift = 10 - k;  // e.g., if numBins = 8, then shift = 7.

    // Compute flattened thread id.
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;

    // Initialize the per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Each block processes tiles in a grid-stride loop.
    // Each tile has tileSizeInts integers.
    const int tileGlobalSize = gridDim.x * tileSizeInts;
    const int firstOffset = blockIdx.x * tileSizeInts;

    // --- Load the first tile into tile0 ---
    {
        int globalIndex = firstOffset + tid * intsPerThread;
        if (globalIndex + intsPerThread - 1 < N) {
            // Load via vectorized int4 loads (8 iterations * 4 ints = 32 ints)
            #pragma unroll
            for (int i = 0; i < intsPerThread / 4; i++) {
                int4 tmp = ((const int4*)data)[(globalIndex + i * 4) / 4];
                int base = tid * intsPerThread + i * 4;
                tile0[base + 0] = tmp.x;
                tile0[base + 1] = tmp.y;
                tile0[base + 2] = tmp.z;
                tile0[base + 3] = tmp.w;
            }
        } else {
            // If there aren’t enough elements, load one int at a time.
            for (int i = 0; i < intsPerThread; i++) {
                int idx = globalIndex + i;
                tile0[tid * intsPerThread + i] = (idx < N) ? data[idx] : -1;
            }
        }
    }
    __syncthreads();

    // --- Process subsequent tiles using double buffering ---
    for (int offset = firstOffset + tileGlobalSize; offset < N; offset += tileGlobalSize) {
        // Load next tile into tile1.
        int globalIndex = offset + tid * intsPerThread;
        if (globalIndex + intsPerThread - 1 < N) {
            #pragma unroll
            for (int i = 0; i < intsPerThread / 4; i++) {
                int4 tmp = ((const int4*)data)[(globalIndex + i * 4) / 4];
                int base = tid * intsPerThread + i * 4;
                tile1[base + 0] = tmp.x;
                tile1[base + 1] = tmp.y;
                tile1[base + 2] = tmp.z;
                tile1[base + 3] = tmp.w;
            }
        } else {
            for (int i = 0; i < intsPerThread; i++) {
                int idx = globalIndex + i;
                tile1[tid * intsPerThread + i] = (idx < N) ? data[idx] : -1;
            }
        }
        __syncthreads();

        // Process the current tile in tile0.
        {
            // Each thread processes its own contiguous block of ints.
            int start = tid * intsPerThread;
            int end   = start + intsPerThread;
            int localBin = -1;
            int localCount = 0;
            for (int i = start; i < end; i++) {
                int value = tile0[i];
                if (value < 0) continue;
                int bin = value >> shift;
                if (bin == localBin) {
                    localCount++;
                } else {
                    if (localCount > 0)
                        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                    localBin = bin;
                    localCount = 1;
                }
            }
            if (localCount > 0)
                atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
        }
        __syncthreads();

        // Swap tile buffers so that tile1 becomes the current tile.
        int *tempPtr = tile0;
        tile0 = tile1;
        tile1 = tempPtr;
        __syncthreads();
    }

    // --- Process the final tile loaded in tile0 ---
    {
        int start = tid * intsPerThread;
        int end   = start + intsPerThread;
        int localBin = -1;
        int localCount = 0;
        for (int i = start; i < end; i++) {
            int value = tile0[i];
            if (value < 0) continue;
            int bin = value >> shift;
            if (bin == localBin) {
                localCount++;
            } else {
                if (localCount > 0)
                    atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                localBin = bin;
                localCount = 1;
            }
        }
        if (localCount > 0)
            atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
    }
    __syncthreads();

    // --- Reduce per-warp histograms into a block-level (partial) histogram ---
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
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    // Note: With fixed block dimensions (8x32), total threads per block is 256.
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // With fixed block dimensions (8 x 32), total threads per block is 8*32 = 256.
    const int blockSizeTotal = 8 * 32;
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    // Set fixed block dimensions: 8 x 32.
    dim3 block(8, 32);
    dim3 grid(gridSize);
    
    // Calculate shared memory size:
    // Two tile buffers: each tile holds block.x * block.y * 32 integers,
    // plus per-warp histograms: (block.x * block.y / 32) * numBins integers.
    int tileSizeInts = block.x * block.y * 32;
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
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSizeTotal, sharedMemSize);
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;
    printf("Occupancy per SM: %f %%\n", occupancy);
    
    // Display device properties.
    int coresPerSM = 64;
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    double clockHz = deviceProp.clockRate * 1000.0;
    double theoreticalOps = totalCores * clockHz * 2;
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", totalCores);
    printf("Clock Rate: %0.2f GHz\n", clockHz / 1e9);
    printf("Theoretical Peak Ops/sec (int): %e ops/sec\n", theoreticalOps);
    
    // Copy final histogram from device to host.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    
    // Print all histogram bins.
    printf("Final Histogram Bins:\n");
    for (int i = 0; i < numBins; i++) {
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