#ifndef HISTOGRAM_TILED_CUH
#define HISTOGRAM_TILED_CUH

#include <cuda.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "utils.cuh"
namespace cg = cooperative_groups;

// Optimized histogram kernel (simplified version):
// - Removes vectorized loads and double buffering.
// - Uses a grid‑stride loop to process one int at a time.
// - Computes a per‑warp histogram in shared memory and then reduces to a block‑level partial histogram.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory is used solely for per-warp histograms.
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    int blockThreads = blockDim.x * blockDim.y;
    int numWarps = blockThreads / warpSize;
    // Use all shared memory for per-warp histogram: we need (numWarps * numBins) integers.
    int *warpHist = sharedMem;

    // Precompute the bit shift factor.
    // Since 1024 is the maximum value and numBins is 2^k, each bin covers 1024/numBins values.
    // This shift equals: log2(1024) - log2(numBins) = 10 - log2(numBins).
    int k = 0;
    int temp = numBins;
    while (temp > 1) {
        k++;
        temp >>= 1;
    }
    int shift = 10 - k;  // e.g., if numBins = 8 (k = 3), shift = 7.

    // Flatten the thread index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // Initialize per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Compute total threads in the grid (over blocks in X dimension).
    int totalThreads = blockThreads * gridDim.x;
    // Compute global starting index for the current thread.
    int globalId = blockIdx.x * blockThreads + tid;

    // Process data via a grid-stride loop. Loads one int at a time.
    for (int i = globalId; i < N; i += totalThreads) {
        int value = data[i];
        // Map the data value from [0,1023] into [0, numBins-1] using a bit-shift.
        int bin = value >> shift;
        if (bin >= 0 && bin < numBins) {
            atomicAdd(&warpHist[warp_id * numBins + bin], 1);
        }
    }
    __syncthreads();

    // Reduce per-warp histograms into a block-level partial histogram.
    // Let only one warp (warp_id==0) perform the final reduction.
    if (warp_id == 0) {
        for (int b = lane; b < numBins; b += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + b];
            }
            partialHist[blockIdx.x * numBins + b] = sum;
        }
    }
}

#endif // HISTOGRAM_OPTIMIZED_SIMPLE_CUH