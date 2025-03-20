#ifndef HISTOGRAM_NAIVE_CUH
#define HISTOGRAM_NAIVE_CUH

#include <cuda.h>
#include <cstdio>
#include "utils.cuh"

// naive approach
__global__ void histogram_naive_kernel(const int *data, int *finalHist, int N, int numBins) {
    extern __shared__ int sharedHist[];
    // indexing
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int blockThreads = blockDim.x * blockDim.y;
    
    // shared histogram
    for (int b = tid; b < numBins; b += blockThreads) {
        sharedHist[b] = 0;
    }
    __syncthreads();
    
    // grid stride loop
    int globalId = (blockIdx.x * blockThreads) + tid;
    int stride = blockThreads * gridDim.x;
    for (int i = globalId; i < N; i += stride) {
        int value = data[i];
        if (value >= 0 && value < numBins) {
            atomicAdd(&sharedHist[value], 1);
        }
    }
    __syncthreads();
    
    // each block updates final histogram
    for (int b = tid; b < numBins; b += blockThreads) {
        atomicAdd(&finalHist[b], sharedHist[b]);
    }
}

#endif // HISTOGRAM_NAIVE_CUH