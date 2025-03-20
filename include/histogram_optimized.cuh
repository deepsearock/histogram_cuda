#ifndef HISTOGRAM_OPTIMIZED_CUH
#define HISTOGRAM_OPTIMIZED_CUH

#include <cuda.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "utils.cuh"
namespace cg = cooperative_groups;

// optimized kernel with double buffering, vectorized memory loads, and per-thread run-length encoding.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // [tile0 | tile1 | per-warp histograms]
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    int blockThreads = blockDim.x * blockDim.y;
    int tileSizeInts = blockThreads * 4;
    int *tile0 = sharedMem;//first tile buffer
    int *tile1 = sharedMem + tileSizeInts;//second tile buffer
    int numWarps = blockThreads / warpSize;        
    int *warpHist = (int*)(sharedMem + 2 * tileSizeInts); //per-warp histogram region

    int k = 0;
    int temp = numBins;
    while (temp > 1) {
        k++;
        temp >>= 1;
    }
    int shift = 10 - k; 


    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // initialize per warp histograms
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // compute global tile size
    int globalTileSizeInts = gridDim.x * tileSizeInts;
    int firstOffset = blockIdx.x * tileSizeInts;

    // use __ldg to load data in vectorized manner
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;

        // call prefetch helper to load data
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

    //double buffering
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        int globalIndex = offset + tid * 4;
        // call prefetch helper to load data
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

        // use per thread aggregation to compute per warp histograms
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

        // swap tiles
        int *tempPtr = tile0;
        tile0 = tile1;
        tile1 = tempPtr;
        __syncthreads();
    }

    // process final tile
    {
        int localHist[256];
        #pragma unroll
        for (int b = 0; b < numBins; b++) {
            localHist[b] = 0;
        }

        for (int i = tid; i < tileSizeInts; i += blockThreads) {
            int value = tile0[i];
            if (value < 0) continue;
            // Compute bin index via bit-shift.
            int bin = value >> shift;
            if(bin < numBins)
                localHist[bin]++;
        }


        // reduce per-thread histograms to per-warp histograms.
        unsigned mask = 0xffffffff; 
        for (int bin = 0; bin < numBins; bin++) {
            int sum = localHist[bin];
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(mask, sum, offset);
            }
            // only lane 0 writes to global memory
            if(lane == 0) {
                atomicAdd(&warpHist[warp_id * numBins + bin], sum);
            }
        }
    }
    __syncthreads();

    // per warp to per block histograms
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


#endif // HISTOGRAM_OPTIMIZED_CUH