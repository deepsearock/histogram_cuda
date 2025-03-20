#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>

#include "../include/utils.cuh"
#include "../include/histogram_naive.cuh"        
#include "../include/histogram_optimized.cuh"    
#include "../include/histogram_tiling.cuh"         

extern __global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks);

using namespace cooperative_groups;

int main(int argc, char *argv[]) {
    // input
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    

    // error handling
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }

    const int blockSizeTotal = 8 * 32; 
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    

    dim3 block(4, 64); 
    dim3 grid(gridSize);

    // calculate shared memory sizes for each kernel.
    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize_optimized = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    size_t sharedMemSize_tiled = (numWarps * numBins) * sizeof(int);
    size_t sharedMemSize_naive = numBins * sizeof(int);
    
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }

    // random data
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;
    }
    // cuda memory allocation and copy
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    

    //naive kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMemset(d_finalHist, 0, finalHistSize);
    cudaEventRecord(start, 0);
    
    histogram_naive_kernel<<<grid, block, sharedMemSize_naive>>>(d_data, d_finalHist, N, numBins);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeNaive;
    cudaEventElapsedTime(&elapsedTimeNaive, start, stop);

    double totalOps = 3.0 * N + ((double)N / tileSizeInts) * numBins;
    double elapsedSecNaive = elapsedTimeNaive / 1000.0;
    double opsPerSecNaive = totalOps / elapsedSecNaive;
    double measuredGopsNaive = opsPerSecNaive / 1e9;
    
    printf("\n=== Naive Kernel ===\n");
    printf("Total execution time: %f ms\n", elapsedTimeNaive);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSecNaive);
    printf("Measured Performance: %f Gops (atomic ops metric)\n", measuredGopsNaive);
    
    // reset memory optimized kernel
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    cudaEventRecord(start, 0);
    
    histogram_optimized_kernel<<<grid, block, sharedMemSize_optimized>>>(d_data, d_partialHist, N, numBins);
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeOptimized;
    cudaEventElapsedTime(&elapsedTimeOptimized, start, stop);
    
    double elapsedSecOptimized = elapsedTimeOptimized / 1000.0;
    double opsPerSecOptimized = totalOps / elapsedSecOptimized;
    double measuredGopsOptimized = opsPerSecOptimized / 1e9;
    
    printf("\n=== Optimized Kernel ===\n");
    printf("Total execution time: %f ms\n", elapsedTimeOptimized);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSecOptimized);
    printf("Measured Performance: %f Gops (atomic ops metric)\n", measuredGopsOptimized);
    
    // reset memory tiled kernel
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    cudaEventRecord(start, 0);

    histogram_tiled_kernel<<<grid, block, sharedMemSize_tiled>>>(d_data, d_partialHist, N, numBins);
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTimeTiled;
    cudaEventElapsedTime(&elapsedTimeTiled, start, stop);
    
    double elapsedSecTiled = elapsedTimeTiled / 1000.0;
    double opsPerSecTiled = totalOps / elapsedSecTiled;
    double measuredGopsTiled = opsPerSecTiled / 1e9;
    
    printf("\n=== Tiled Kernel ===\n");
    printf("Total execution time: %f ms\n", elapsedTimeTiled);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSecTiled);
    printf("Measured Performance: %f Gops (atomic ops metric)\n", measuredGopsTiled);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSizeTotal, sharedMemSize_optimized);
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;
    printf("\nOccupancy per SM (Optimized Kernel): %f %%\n", occupancy);

    int coresPerSM = 64;
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    double clockHz = deviceProp.clockRate * 1000.0;
    double theoreticalOps = totalCores * clockHz * 2;
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", totalCores);
    printf("Clock Rate: %0.2f GHz\n", clockHz / 1e9);
    printf("Theoretical Peak Ops/sec (int): %e ops/sec\n", theoreticalOps);


    // histogram check
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    printf("\nFinal Histogram (nonzero bins):\n");
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Bin %d: %d\n", i, h_finalHist[i]);
    }

    free(h_data);
    free(h_finalHist);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}