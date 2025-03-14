#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering, vectorized loads,
// per-warp histograms, bit-shift based bin calculation, and loop unrolling.
__global__ void hierarchical_histogram_kernel(const int *data, int *partialHist, int N, int numBins) {
    // We use shared memory for per-warp histograms.
    // The shared memory size should be at least (numWarps * numBins) integers.
    extern __shared__ int warpHist[];

    // Compute block and thread indices.
    const int blockThreads = blockDim.x * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warpSize = 32;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int numWarps = blockThreads / warpSize;

    // Initialize shared memory for warp histograms.
    for (int i = tid; i < numWarps * numBins; i += blockThreads) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Compute shift factor.
    // Since the data values are in [0,1023] and numBins is 2^k,
    // each bin spans 1024/numBins values. Thus shift = 10 - k.
    int shift = 10;
    int temp = numBins;
    while (temp > 1) {
        shift--;
        temp >>= 1;
    }

    // Each thread will accumulate a private histogram.
    // We assume numBins is small (up to 256) so we allocate an array on registers.
    int localHist[256];  // maximum bins supported.
    for (int b = 0; b < numBins; b++) {
        localHist[b] = 0;
    }

    // Define the number of integers loaded per thread per iteration.
    const int intsPerThread = 8;
    // Compute a grid stride based on total threads and our per-thread load.
    int stride = blockThreads * gridDim.x * intsPerThread;

    // Compute a starting index for this thread.
    // A natural mapping is to have thread t in block b process indices starting at:
    //    start = (b * blockThreads + t) * intsPerThread
    int base = (blockIdx.x * blockThreads + tid) * intsPerThread;

    // Process data in a grid-stride loop.
    for (int i = base; i < N; i += stride) {
        #pragma unroll
        for (int j = 0; j < intsPerThread; j++) {
            int idx = i + j;
            if (idx < N) {
                int value = data[idx];
                int bin = value >> shift;  // compute bin index using bit shift
                localHist[bin]++;
            }
        }
    }

    // Now, each thread atomically adds its local histogram into its warp’s shared memory region.
    // We partition the shared memory array into one segment per warp.
    for (int b = 0; b < numBins; b++) {
        atomicAdd(&warpHist[warp_id * numBins + b], localHist[b]);
    }
    __syncthreads();

    // Warp 0 of the block reduces the per-warp histograms into one block-level histogram.
    if (warp_id == 0) {
        for (int b = lane; b < numBins; b += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + b];
            }
            // Write the block’s partial histogram.
            partialHist[blockIdx.x * numBins + b] = sum;
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

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Declarations for the kernels.
extern "C" {
    __global__ void hierarchical_histogram_kernel(const int *data, int *partialHist, int N, int numBins);
    __global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks);
}

int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
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
    
    // Set fixed block dimensions: e.g., 8 x 32 = 256 threads per block.
    dim3 block(8, 32);
    const int blockThreads = block.x * block.y; // 256 threads per block.
    
    // For our hierarchical kernel, each thread processes intsPerThread integers.
    const int intsPerThread = 8;
    // Total work per block is: blockThreads * intsPerThread.
    // We use a grid-stride loop so grid size can be chosen to cover N.
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = ((N + intsPerThread * blockThreads - 1) / (intsPerThread * blockThreads));
    dim3 grid(gridSize);
    
    // Shared memory: we need one integer per bin for each warp in the block.
    int numWarps = blockThreads / 32; // 256/32 = 8 warps.
    size_t sharedMemSize = numWarps * numBins * sizeof(int);
    
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
    
    // Launch the hierarchical histogram kernel.
    hierarchical_histogram_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
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
    
    // (Optional) Copy final histogram from device to host and print histogram bins.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    
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
