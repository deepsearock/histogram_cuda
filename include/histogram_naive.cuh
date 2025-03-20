#ifndef HISTOGRAM_NAIVE_CUH
#define HISTOGRAM_NAIVE_CUH

#include <cuda.h>
#include <cstdio>
#include "utils.cuh"


// naive approach 
__global__ void histogram_naive_kernel(const int *data, int *finalHist, int N, int numBins) {
    extern __shared__ int sharedHist[];
    int tid = threadIdx.x;
    int blockThreads = blockDim.x;

    // shared histogram
    for (int b = tid; b < numBins; b += blockThreads) {
        sharedHist[b] = 0;
    }
    __syncthreads();
    
    // grid stride loop
    int globalId = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for (int i = globalId; i < N; i += stride) {
        int value = data[i];
        if (value >= 0 && value < numBins) {
            atomicAdd(&sharedHist[value], 1);
        }
    }
    __syncthreads();
    
    // each block updates the final histogram
    for (int b = tid; b < numBins; b += blockThreads) {
        atomicAdd(&finalHist[b], sharedHist[b]);
    }
}

#endif // HISTOGRAM_NAIVE_CUH