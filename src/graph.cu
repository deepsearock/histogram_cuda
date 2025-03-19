#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>

#include "../include/utils.cuh"
#include "../include/histogram_naive.cuh"        // Kernel: histogram_naive_kernel(const int*, int*, int, int)
#include "../include/histogram_optimized.cuh"    // Kernel: histogram_optimized_kernel(const int*, int*, int, int)
#include "../include/histogram_tiling.cuh"         // Kernel: histogram_tiled_kernel(const int*, int*, int, int)
// Reduction kernel used for optimized and tiled kernels.
extern __global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks);

using namespace cooperative_groups;

int main(int argc, char *argv[]) {
    // Usage: ./graph -i <VecDim> [GridSize]
    if (argc < 3 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[2]);    // vector dimension / number of elements
    int gridSize;
    const int blockSizeTotal = 8 * 32; // 256 threads per block
    if (argc >= 4)
        gridSize = atoi(argv[3]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    // Set block and grid dimensions.
    dim3 block(4, 64);   // 4 x 64 = 256 threads per block.
    dim3 grid(gridSize);
    
    // Shared memory sizes for each kernel:
    // Naive kernel: one histogram per block.
    size_t sharedMemSize_naive = 0; // no extra shared mem allocated, as the kernel allocates histogram of 'numBins' internally.
    // For optimized kernel:
    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize_optimized = (2 * tileSizeInts + numWarps * /*numBins*/256) * sizeof(int);
    // For tiled kernel: only per-warp histogram.
    size_t sharedMemSize_tiled = (numWarps * /*numBins*/256) * sizeof(int);
    // (Note: These shared mem sizes will be adjusted inside the loop below based on numBins.)
    
    // Data sizes.
    size_t dataSize = N * sizeof(int);
    
    // Allocate and initialize host data.
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024; // values in [0, 1023]
    }
    
    // Allocate device memory.
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    // We'll allocate d_partialHist and d_finalHist based on maximum possible sizes.
    // Maximum partialHist is gridSize * max(numBins) integers.
    int maxBins = 256;
    size_t partialHistSize = gridSize * maxBins * sizeof(int);
    size_t finalHistSize = maxBins * sizeof(int);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    
    // Copy input data to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    
    // Open CSV file for writing results.
    FILE *fp = fopen("results.csv", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open CSV file for writing.\n");
        return 1;
    }
    // Write CSV header.
    fprintf(fp, "Kernel,NumBins,ExecutionTime_ms,TotalOps,Throughput_ops_sec,Gops\n");
    
    // Define the bin sizes to test.
    const int binSizes[7] = {4, 8, 16, 32, 64, 128, 256};
    // Calculate a common total operations estimate.
    double totalOps = 3.0 * N + ((double)N / (block.x * block.y * 4)) * maxBins; // using maxBins for estimation
    
    // Create CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // For each kernel variant, loop through each bin setting.
    
    // --- Naive Kernel ---
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];
        // Reset final histogram.
        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        // Launch naive kernel.
        // Note: histogram_naive_kernel writes directly to finalHist,
        // so we pass numBins and use sharedMemSize_naive as configured in the kernel.
        histogram_naive_kernel<<<grid, block, numBins * sizeof(int)>>>(d_data, d_finalHist, N, numBins);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        double elapsedSec = elapsedTime / 1000.0;
        double opsPerSec = totalOps / elapsedSec;
        double measuredGops = opsPerSec / 1e9;
        
        fprintf(fp, "Naive,%d,%f,%.0f,%e,%f\n", numBins, elapsedTime, totalOps, opsPerSec, measuredGops);
    }
    
    // --- Optimized Kernel ---
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];
        // Adjust shared memory size for optimized kernel.
        size_t sharedMemSizeOpt = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
        // Reset device buffers.
        cudaMemset(d_partialHist, 0, gridSize * numBins * sizeof(int));
        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        // Launch optimized kernel.
        histogram_optimized_kernel<<<grid, block, sharedMemSizeOpt>>>(d_data, d_partialHist, N, numBins);
        // Launch reduction kernel.
        int reduceBlockSize = 256;
        int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
        histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        double elapsedSec = elapsedTime / 1000.0;
        double opsPerSec = totalOps / elapsedSec;
        double measuredGops = opsPerSec / 1e9;
        
        fprintf(fp, "Optimized,%d,%f,%.0f,%e,%f\n", numBins, elapsedTime, totalOps, opsPerSec, measuredGops);
    }
    
    // --- Tiled Kernel ---
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];
        // Adjust shared memory size for tiled kernel.
        size_t sharedMemSizeTile = (numWarps * numBins) * sizeof(int);
        // Reset device buffers.
        cudaMemset(d_partialHist, 0, gridSize * numBins * sizeof(int));
        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        // Launch tiled kernel.
        histogram_tiled_kernel<<<grid, block, sharedMemSizeTile>>>(d_data, d_partialHist, N, numBins);
        // Launch reduction kernel.
        int reduceBlockSize = 256;
        int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
        histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        double elapsedSec = elapsedTime / 1000.0;
        double opsPerSec = totalOps / elapsedSec;
        double measuredGops = opsPerSec / 1e9;
        
        fprintf(fp, "Tiled,%d,%f,%.0f,%e,%f\n", numBins, elapsedTime, totalOps, opsPerSec, measuredGops);
    }
    
    fclose(fp);
    
    // Clean up.
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}