#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel using per-warp private histograms and register-level accumulation.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Use dynamic shared memory for per-warp histograms.
    // Each warp gets numBins integers.
    extern __shared__ int warpHist[]; // size = (numWarps * numBins)
    
    const int warpSize = 32;
    // Flatten 2D thread index: blockDim.x is set to 32.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockThreads = blockDim.x * blockDim.y;
    int numWarps = blockThreads / warpSize;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Each warp initializes its portion of shared memory.
    for (int i = lane; i < numBins; i += warpSize) {
        warpHist[warp_id * numBins + i] = 0;
    }
    __syncthreads();  // Ensure all per-warp histograms are zeroed.

    // Compute global index using flattened thread index.
    int global_tid = blockIdx.x * blockThreads + tid;
    int stride = gridDim.x * blockThreads;

    // Register-level accumulation.
    int localBin = -1;
    int localCount = 0;
    while (global_tid < N) {
        int value = data[global_tid];
        // Map value in [0, 1023] to bin [0, numBins-1].
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
        global_tid += stride;
    }
    // Flush any remaining count.
    if (localCount > 0) {
        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
    }
    __syncthreads();

    // Reduce per-warp histograms into a single block-level (partial) histogram.
    // Let threads in the first warp perform the reduction.
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

// Reduction kernel: sum the partial histograms into the final histogram.
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
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8.\n");
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
    // Use 1D grid for now.
    dim3 grid(gridSize);

    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);

    // Allocate and initialize host memory.
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;
    }

    // Allocate device memory.
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);

    // Copy input data.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);

    // Create CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    // Launch the optimized histogram kernel.
    // Compute shared memory size: (numWarps * numBins) integers.
    int blockThreads = block.x * block.y;
    int numWarps = blockThreads / 32;
    size_t sharedMemSize = numWarps * numBins * sizeof(int);
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
    double totalOps = (double) N + (gridSize * numBins);
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
    // Compute occupancy as the fraction of active threads relative to the maximum.
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;  // percentage
    printf("Occupancy per SM: %f %%\n", occupancy);

    // Copy final histogram back to host.
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
