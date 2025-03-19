#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "histogram_naive.cuh"
#include "histogram_optimized.cuh"
#include "histogram_tiling.cuh"

int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // With fixed block dimensions (example: 4 x 64 = 256 threads per block).
    const int blockSizeTotal = 4 * 64;
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    // Set fixed block and grid dimensions.
    dim3 block(4, 64);
    dim3 grid(gridSize);
    
    // Shared memory size calculation for the optimized kernel:
    // Two tile buffers: tileSizeInts = block.x * block.y * 4 ints per tile.
    // Plus per-warp histogram: numWarps * numBins integers.
    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    
    // Data sizes.
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    
    // Allocate and initialize host data.
    int *h_data = (int *)malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;  // values in [0, 1023]
    }
    
    // Allocate device memory.
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void **)&d_data, dataSize);
    cudaMalloc((void **)&d_partialHist, partialHistSize);
    cudaMalloc((void **)&d_finalHist, finalHistSize);
    
    // Copy input data to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    
    // --- Run Naive Kernel ---
    // For the naive kernel, shared memory size is numBins*sizeof(int)
    cudaMemset(d_finalHist, 0, finalHistSize);
    PerfMetrics naivePerf = runNaiveHistogram(d_data, d_finalHist, N, numBins, gridSize, block, grid, numBins * sizeof(int));
    printf("Naive Kernel - Bins = %d, Time = %.3f ms, GFLOPS = %.3f\n", numBins, naivePerf.ms, naivePerf.Gops);
    
    // --- Run Tiled Kernel ---
    // Clear device histograms.
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    PerfMetrics tiledPerf = runTiledHistogram(d_data, d_finalHist, N, numBins, gridSize, block, grid, numWarps * numBins * sizeof(int));
    printf("Tiled Kernel - Bins = %d, Combined Time = %.3f ms, GFLOPS = %.3f\n", numBins, tiledPerf.ms, tiledPerf.Gops);
    
    // --- Run Optimized Kernel ---
    // Clear device histograms.
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    PerfMetrics optimizedPerf = runOptimizedHistogram(d_data, d_finalHist, N, numBins, gridSize, block, grid, sharedMemSize);
    printf("Optimized Kernel - Bins = %d, Combined Time = %.3f ms, GFLOPS = %.3f\n", numBins, optimizedPerf.ms, optimizedPerf.Gops);

    // (Optional) Validate final histogram (for one kernel, e.g., optimized).
    int *h_finalHist = (int *)malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Final Histogram Bin %d: %d\n", i, h_finalHist[i]);
    }
    
    // Clean up.
    free(h_data);
    free(h_finalHist);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    
    return 0;
}