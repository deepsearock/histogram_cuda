#ifndef HISTOGRAM_NAIVE_CUH
#define HISTOGRAM_NAIVE_CUH

#include <cuda.h>
#include <cstdio>
#include "utils.cuh"

// Naive histogram kernel: Each block builds a shared memory histogram
// and atomically updates the final histogram in global memory.
__global__ void histogram_naive_kernel(const int *data, int *finalHist, int N, int numBins) {
    extern __shared__ int sharedHist[];
    int tid = threadIdx.x;
    int blockThreads = blockDim.x;

    // Initialize the shared histogram.
    for (int b = tid; b < numBins; b += blockThreads) {
        sharedHist[b] = 0;
    }
    __syncthreads();
    
    // Process input data with a grid-stride loop.
    int globalId = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for (int i = globalId; i < N; i += stride) {
        int value = data[i];
        if (value >= 0 && value < numBins) {
            atomicAdd(&sharedHist[value], 1);
        }
    }
    __syncthreads();
    
    // Each block atomically updates the final global histogram.
    for (int b = tid; b < numBins; b += blockThreads) {
        atomicAdd(&finalHist[b], sharedHist[b]);
    }
}

// Host-callable function to launch the naive histogram kernel.
// Note: The shared memory size should be at least numBins*sizeof(int).
// This version returns the measured performance.
__host__ PerfMetrics runNaiveHistogram(const int *d_data, int *d_finalHist, int N, int numBins, 
                                         int gridSize, dim3 block, dim3 grid, size_t sharedMemSize) {
    // Clear the final histogram on the device.
    cudaMemset(d_finalHist, 0, numBins * sizeof(int));
    
    // Approximate total operations for the naive kernel.
    double totalOpsNaive = static_cast<double>(N);
    
    PerfMetrics metricsNaive = measureKernelPerformance(grid, block, sharedMemSize,
                                                         totalOpsNaive, histogram_naive_kernel,
                                                         d_data, d_finalHist, N, numBins);
    // Print measured performance.
    printf("Naive Histogram Kernel: Bins = %d, Time = %.3f ms, Gops = %.3f\n",
           numBins, metricsNaive.ms, metricsNaive.Gops);
    
    return metricsNaive;
}

#endif // HISTOGRAM_NAIVE_CUH