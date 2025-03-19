#ifndef HISTOGRAM_OPTIMIZED_CUH
#define HISTOGRAM_OPTIMIZED_CUH

#include <cuda.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "utils.cuh"
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering and aggressive reduction.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory layout:
    // [tile0 | tile1 | per-warp histograms]
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    int blockThreads = blockDim.x * blockDim.y;
    // Each tile holds blockThreads * 4 integers (vectorized loads: int4)
    int tileSizeInts = blockThreads * 4;
    int *tile0 = sharedMem;                      // first tile buffer
    int *tile1 = sharedMem + tileSizeInts;         // second tile buffer
    int numWarps = blockThreads / warpSize;        // assume blockThreads is a multiple of 32
    int *warpHist = (int*)(sharedMem + 2 * tileSizeInts); // per-warp histogram region

    // Precompute the bit-shift factor.
    // Since 1024 is the max value and numBins is 2^k, each bin spans 1024/numBins values.
    // log2(1024/numBins) = 10 - log2(numBins)
    int k = 0;
    int temp = numBins;
    while (temp > 1) {
        k++;
        temp >>= 1;
    }
    int shift = 10 - k;  // e.g., if numBins = 8 (k=3), then shift = 7.

    // Flatten the thread index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Initialize per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Compute global tile size (in ints) for double buffering.
    int globalTileSizeInts = gridDim.x * tileSizeInts;
    int firstOffset = blockIdx.x * tileSizeInts;

    // Load the first tile from global memory into tile0 using __ldg and prefetching.
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;
        // Prefetch future data (64 is an arbitrary offset; adjust as needed)
        prefetch_global(&data[globalIndex + 128]);
        if (globalIndex + 3 < N) {
            int4 tmp = __ldg(reinterpret_cast<const int4*>(&data[globalIndex]));
            tile0[tid * 4 + 0] = tmp.x;
            tile0[tid * 4 + 1] = tmp.y;
            tile0[tid * 4 + 2] = tmp.z;
            tile0[tid * 4 + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile0[tid * 4 + i] = (idx < N) ? __ldg(&data[idx]) : -1;
            }
        }
    }
    __syncthreads();

    // Process the tiles using double buffering.
    // In the double-buffering loop, prefetch the next tile and load using __ldg.
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        int globalIndex = offset + tid * 4;
        // Prefetch a future block of data. Adjust "64" as a prefetch distance.
        prefetch_global(&data[globalIndex + 128]);
        if (globalIndex + 3 < N) {
            int4 tmp = __ldg(reinterpret_cast<const int4*>(&data[globalIndex]));
            tile1[tid * 4 + 0] = tmp.x;
            tile1[tid * 4 + 1] = tmp.y;
            tile1[tid * 4 + 2] = tmp.z;
            tile1[tid * 4 + 3] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile1[tid * 4 + i] = (idx < N) ? __ldg(&data[idx]) : -1;
            }
        }
        __syncthreads();

        // Process the current tile (in tile0) using a per-thread run-length aggregation.
        {
            int localBin = -1;
            int localCount = 0;
            #pragma unroll
            for (int i = tid; i < tileSizeInts; i += blockThreads) {
                int value = tile0[i];
                if (value < 0) continue;
                // Use bit shift instead of division.
                int bin = value >> shift;
                if (bin == localBin) {
                    localCount++;
                } else {
                    if (localCount > 0)
                        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                    localBin = bin;
                    localCount = 1;
                }
            }
            if (localCount > 0)
                atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
        }
        __syncthreads();

        // Swap tile buffers.
        int *tempPtr = tile0;
        tile0 = tile1;
        tile1 = tempPtr;
        __syncthreads();
    }

    // Process the final tile loaded in tile0.
    {
        // Assume numBins is small enough; here we allocate a fixed-size array for local counts.
        // For maximum flexibility, we allocate up to 256 entries (since k ∈ [2,8]).
        int localHist[256];
        #pragma unroll
        for (int b = 0; b < numBins; b++) {
            localHist[b] = 0;
        }

        // Each thread processes multiple elements from the current tile.
        for (int i = tid; i < tileSizeInts; i += blockThreads) {
            int value = tile0[i];
            if (value < 0) continue;
            // Compute bin index via bit-shift.
            int bin = value >> shift;
            if(bin < numBins)
                localHist[bin]++;
        }

        // For each bin, perform a warp-level reduction using shuffles so that only one thread per warp
        // issues an atomicAdd to the per-warp histogram.
        unsigned mask = 0xffffffff;  // Full warp
        for (int bin = 0; bin < numBins; bin++) {
            int sum = localHist[bin];
            // Reduce across the warp.
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset);
            }
            // Only lane 0 in each warp contributes to the global per-warp histogram.
            if(lane == 0) {
                atomicAdd(&warpHist[warp_id * numBins + bin], sum);
            }
        }
    }
    __syncthreads();

    // Reduce the per-warp histograms into a block-level (partial) histogram.
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

// Reduction kernel: Sum partial histograms from all blocks into the final histogram.

#endif // HISTOGRAM_OPTIMIZED_CUH