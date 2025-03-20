#ifndef HISTOGRAM_TILING_CUH
#define HISTOGRAM_TILING_CUH

#include <cuda.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "utils.cuh"
namespace cg = cooperative_groups;

#define WARP_SIZE 32

//tiled strategy
__global__ void histogram_tiled_kernel(const int *data, int *partialHist, int N, int numBins) {
    // indexing
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int blockThreads = blockDim.x * blockDim.y;
    int numWarps = blockThreads / WARP_SIZE;
    

    //allocate shared memory
    extern __shared__ int sharedWarpHist[];
    
    for (int b = lane; b < numWarps * numBins; b += WARP_SIZE) {
        sharedWarpHist[b] = 0;
    }
    __syncthreads();
    
    // grid stride loop
    int globalId = blockIdx.x * blockThreads + tid;
    int stride = gridDim.x * blockThreads;
    for (int i = globalId; i < N; i += stride) {
        int value = data[i];
        // map data
        int bin = (value * numBins) / 1024;
        if (bin >= 0 && bin < numBins) {
            atomicAdd(&sharedWarpHist[warp_id * numBins + bin], 1);
        }
    }
    __syncthreads();
    
    // reduce warp level histogram to block level
    if (warp_id == 0) {
        for (int b = lane; b < numBins; b += WARP_SIZE) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += sharedWarpHist[w * numBins + b];
            }
            partialHist[blockIdx.x * numBins + b] = sum;
        }
    }
}

#endif // HISTOGRAM_TILING_CUH