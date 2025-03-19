#ifndef HISTOGRAM_OPTIMIZED_CUH
#define HISTOGRAM_OPTIMIZED_CUH

#include <cuda.h>
#include <cooperative_groups.h>
#include <cstdio>
#include "utils.cuh"
namespace cg = cooperative_groups;

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

// Optimized histogram kernel with double buffering and aggressive reduction.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory layout with padding:
    // [tile0 (tileSizeInts ints) + padTile] | [tile1 (tileSizeInts ints) + padTile] |
    // [per-warp histograms with row stride = numBins + padHist]
    extern __shared__ int sharedMem[];
    const int warpSize = 32;
    int blockThreads = blockDim.x * blockDim.y;
    // Each tile holds blockThreads * 4 integers (vectorized loads via int4).
    int tileSizeInts = blockThreads * 4;
    // Padding values to reduce bank conflicts.
    int padTile = 1;
    int padHist = 1;
    // Tile buffer sizes include padding.
    int tile0Size = tileSizeInts + padTile;
    int tile1Size = tileSizeInts + padTile;
    
    int *tile0 = sharedMem;                      // first tile buffer
    int *tile1 = sharedMem + tile0Size;          // second tile buffer
    // Per-warp histograms: each warp gets (numBins + padHist) integers.
    int *warpHist = sharedMem + tile0Size + tile1Size;
    
    int numWarps = blockThreads / warpSize; // Assuming blockThreads is a multiple of warpSize.
    
    // Precompute the bit-shift factor.
    // Since 1024 is the max value and numBins is 2^k, each bin spans 1024/numBins values:
    // shift = 10 - log2(numBins)
    int k = 0;
    int temp = numBins;
    while (temp > 1) {
        k++;
        temp >>= 1;
    }
    int shift = 10 - k;
    
    // Flatten thread index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    
    // Initialize per-warp histogram with padding per row.
    for (int i = lane; i < numWarps * (numBins + padHist); i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();
    
    // Compute global tile size (in ints) for double buffering.
    int globalTileSizeInts = gridDim.x * tileSizeInts;
    int firstOffset = blockIdx.x * tileSizeInts;
    
    // Load the first tile from global memory into tile0.
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;
        prefetch_global(&data[globalIndex + 128]);  // Prefetch with an offset (adjust as needed).
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
    
    // Process tiles using double buffering.
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        int globalIndex = offset + tid * 4;
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
        
        // Process the current tile (in tile0) with aggressive warp-level reduction.
        {
            // Allocate a register-based local histogram.
            int localHist[256]; // Maximum numBins assumed to be <= 256.
            #pragma unroll
            for (int b = 0; b < numBins; b++) {
                localHist[b] = 0;
            }
            // Each thread processes its portion of tile0.
            for (int i = tid; i < tileSizeInts; i += blockThreads) {
                int value = tile0[i];
                if (value < 0) continue;
                int bin = value >> shift;  // Faster than division.
                if (bin < numBins)
                    localHist[bin]++;
            }
            // For each bin, reduce values across the warp.
            for (int b = 0; b < numBins; b++) {
                int sum = warpReduceSum(localHist[b]);
                if (lane == 0) {
                    atomicAdd(&warpHist[warp_id * (numBins + padHist) + b], sum);
                }
            }
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
        int localHist[256];
        #pragma unroll
        for (int b = 0; b < numBins; b++) {
            localHist[b] = 0;
        }
        for (int i = tid; i < tileSizeInts; i += blockThreads) {
            int value = tile0[i];
            if (value < 0) continue;
            int bin = value >> shift;
            if (bin < numBins)
                localHist[bin]++;
        }
        for (int b = 0; b < numBins; b++) {
            int sum = warpReduceSum(localHist[b]);
            if (lane == 0) {
                atomicAdd(&warpHist[warp_id * (numBins + padHist) + b], sum);
            }
        }
    }
    __syncthreads();
    
    // Reduce per-warp histograms into a block-level (partial) histogram.
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * (numBins + padHist) + i];
            }
            partialHist[blockIdx.x * numBins + i] = sum;
        }
    }
}

// Reduction kernel: Sum partial histograms from all blocks into the final histogram.


// Host-callable function to launch the histogram kernels.
// This function allocates the required partial histogram buffer,
// launches the optimized kernel and the reduction kernel, and then frees temporary memory.
__host__ PerfMetrics runOptimizedHistogram(const int *d_data, int *d_finalHist, int N, int numBins, 
    int gridSize, dim3 block, dim3 grid, size_t sharedMemSize) {
size_t partialHistSize = gridSize * numBins * sizeof(int);
int *d_partialHist = nullptr;
cudaMalloc((void**)&d_partialHist, partialHistSize);
cudaMemset(d_partialHist, 0, partialHistSize);

// Approximate total operations for the optimized kernel.
double totalOpsOptimized = static_cast<double>(N);

// Measure the optimized histogram kernel performance.
PerfMetrics metricsOptimized = measureKernelPerformance(grid, block, sharedMemSize, totalOpsOptimized,
                     histogram_optimized_kernel, d_data, d_partialHist, N, numBins);
// Print number of bins and the optimized kernel performance.
printf("Optimized Histogram Kernel: Bins = %d, Time = %.3f ms, Gops = %.3f\n",
numBins, metricsOptimized.ms, metricsOptimized.Gops);

// Launch reduction kernel using measureKernelPerformance.
int reduceBlockSize = 256;
int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;

// Approximate total operations for the reduction kernel.
double totalOpsReduction = static_cast<double>(gridSize * numBins);

PerfMetrics metricsReduction = measureKernelPerformance(dim3(reduceGridSize, 1, 1), dim3(reduceBlockSize, 1, 1), 0, totalOpsReduction,
                     histogram_reduce_kernel, d_partialHist, d_finalHist, numBins, gridSize);
// Print number of bins and the reduction kernel performance.
printf("Reduction Kernel: Bins = %d, Time = %.3f ms, Gops = %.3f\n",
numBins, metricsReduction.ms, metricsReduction.Gops);

// Combine the performance metrics
PerfMetrics combined;
combined.ops = metricsOptimized.ops + metricsReduction.ops;
combined.ms = metricsOptimized.ms + metricsReduction.ms;
combined.opsPerSec = combined.ops / (combined.ms / 1000.0);
combined.Gops = combined.opsPerSec / 1e9;

cudaFree(d_partialHist);

return combined;
}

#endif // HISTOGRAM_OPTIMIZED_CUH