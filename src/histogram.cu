#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel using shared memory for input tiling and per-warp histograms.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // We will use dynamic shared memory for two purposes:
    // 1. An input tile: each block loads tileSize elements (tileSize = blockDim.x * blockDim.y).
    // 2. Per-warp histograms: one histogram (of size numBins) per warp.
    extern __shared__ int sharedMem[];
    int tileSize = blockDim.x * blockDim.y;
    int *tileData = sharedMem; // first tileSize integers
    // The rest of shared memory is used for per-warp histograms.
    int *warpHist = (int*)(sharedMem + tileSize);

    const int warpSize = 32;
    // Flatten the 2D block index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockThreads = tileSize;  // total threads per block
    int numWarps = blockThreads / warpSize;  // assume blockThreads is a multiple of 32
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Initialize the per-warp histogram region.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Total global tile stride (each block processes a portion of the input in tiles).
    int globalTileSize = gridDim.x * tileSize;
    // Each block processes several tiles until its portion of data is done.
    for (int offset = blockIdx.x * tileSize; offset < N; offset += globalTileSize) {
        // Each thread loads one element from global memory into the tile.
        int globalIndex = offset + tid;
        if (globalIndex < N)
            tileData[tid] = data[globalIndex];
        else
            tileData[tid] = -1;  // marker for invalid element
        __syncthreads();

        // Process the tile stored in shared memory.
        // Each thread will process a subset of tileData (using its thread index as starting point).
        int localBin = -1;
        int localCount = 0;
        for (int i = tid; i < tileSize; i += blockThreads) {
            int value = tileData[i];
            if (value < 0) continue;  // skip invalid elements
            // Map value in [0, 1023] to a bin in [0, numBins-1].
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
        // Flush any remaining count.
        if (localCount > 0) {
            atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
        }
        __syncthreads();
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

// Reduction kernel: sums the partial histograms from all blocks into the final histogram.
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
    // Command-line usage:
    // ./histogram_atomic -i <BinNum> <VecDim> [BlockSize] [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize] [GridSize]\n", argv[0]);
        return 1;
    }

    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g. 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }

    // Optionally accept block and grid sizes.
    // BlockSize here refers to total threads per block.
    int blockSizeTotal = 256; // default total threads per block
    int gridSize;
    if (argc >= 5)
        blockSizeTotal = atoi(argv[4]);
    if (argc >= 6)
        gridSize = atoi(argv[5]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;

    // Set 2D block shape: force blockDim.x = 32 and blockDim.y = blockSizeTotal / 32.
    int blockDimX = 32;
    int blockDimY = blockSizeTotal / 32;
    if (blockDimY < 1) blockDimY = 1;
    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridSize);

    // Calculate shared memory size:
    //   For tileData: tileSize = blockDim.x * blockDim.y integers.
    //   For per-warp histogram: numWarps = tileSize/32, so size = numWarps * numBins integers.
    int tileSize = block.x * block.y;
    int numWarps = tileSize / 32;
    size_t sharedMemSize = (tileSize + numWarps * numBins) * sizeof(int);

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

    // Calculate throughput.
    double totalOps = (double) N + (gridSize * numBins); // approximate total atomic operations
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double gflops = opsPerSec / 1e9;

    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
    printf("Performance: %f GFLOPS (atomic ops metric)\n", gflops);

    // Calculate occupancy.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor; // e.g., 2048 for V100
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSizeTotal, sharedMemSize);
    float occupancy = (activeBlocks * blockSizeTotal) / (float)maxThreadsPerSM;
    occupancy = occupancy * 100.0f;  // percentage
    printf("Occupancy per SM: %f %%\n", occupancy);

    // Copy final histogram from device to host.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);

    // Optionally print nonzero bins.
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
