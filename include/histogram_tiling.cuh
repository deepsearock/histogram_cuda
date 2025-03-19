#ifndef HISTOGRAM_TILED_CUH
#define HISTOGRAM_TILED_CUH

#include <cuda.h>
#include <cstdio>
#include "utils.cuh"

#define WARP_SIZE 32
// Tiled histogram kernel:
// Each block computes its partial histogram using per-warp tiling.
// Each warp maintains its own sub-histogram in shared memory. After processing,
// each warp reduces its sub-histogram and atomically accumulates into the block-level partial histogram.
__global__ void histogram_tiled_kernel(const int *data, int *partialHist, int N, int numBins) {
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Allocate shared memory for all warps.
    // We allocate numWarps * numBins ints.
    extern __shared__ int sharedWarpHist[];

    // Initialize each warp's histogram.
    for (int b = lane; b < numBins; b += WARP_SIZE) {
        sharedWarpHist[warp_id * numBins + b] = 0;
    }
    __syncthreads();

    // Process data with a grid-stride loop.
    int globalId = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = globalId; i < N; i += stride) {
        int value = data[i];
        if (value >= 0 && value < numBins) {
            // Each thread atomically updates the histogram for its warp in shared memory.
            atomicAdd(&sharedWarpHist[warp_id * numBins + value], 1);
        }
    }
    __syncthreads();

    // Each warp reduces its histogram rows.
    for (int b = lane; b < numBins; b += WARP_SIZE) {
        // Load the bin count from shared memory.
        int count = sharedWarpHist[warp_id * numBins + b];
        // Perform a warp-level reduction.
        count = warpReduceSum(count);
        // Let one thread per warp (lane 0) update a block-level partial histogram.
        if (lane == 0) {
            // Accumulate contributions from each warp into the same partial histogram slot.
            atomicAdd(&partialHist[blockIdx.x * numBins + b], count);
        }
    }
}

// Host-callable function to launch the tiled histogram kernels.
// This version returns the combined performance metrics from both kernels.
#endif // HISTOGRAM_TILED_CUH