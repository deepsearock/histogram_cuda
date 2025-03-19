#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

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

// Prefetch helper using inline PTX.
__device__ inline void prefetch_global(const void *ptr) {
    asm volatile("prefetch.global.L1 [%0];" :: "l"(ptr));
}

// Warp-level reduction using __shfl_down_sync.
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // UTILS_CUH