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

    //input
    if (argc < 3 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[2]); 
    int gridSize;
    const int blockSizeTotal = 8 * 32; 
    if (argc >= 4)
        gridSize = atoi(argv[3]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    

    dim3 block(4, 64);  
    dim3 grid(gridSize);
    
    // handle errors
    if (N <= gridSize) {
        fprintf(stderr, "Error: VecDim (%d) must be greater than GridSize (%d).\n", N, gridSize);
        return 1;
    }
    

    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;

    size_t dataSize = N * sizeof(int);

    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));

    // random data
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024; 
    }

    // memory
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    int maxBins = 256;
    size_t partialHistSize = gridSize * maxBins * sizeof(int);
    size_t finalHistSize = maxBins * sizeof(int);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);

    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    
    
    //save to file
    char filename[128];
    sprintf(filename, "results-gridsize-%d-vecdim-%d.csv", gridSize, N);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open CSV file %s for writing.\n", filename);
        return 1;
    }

    fprintf(fp, "VecDim,%d,GridSize,%d\n", N, gridSize);

    fprintf(fp, "Kernel_Bins,Gops\n");

    const int binSizes[7] = {4, 8, 16, 32, 64, 128, 256};

    double totalOps = 3.0 * N + ((double)N / (block.x * block.y * 4)) * maxBins; // using maxBins for estimation

    // time event 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // naive kernel
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];

        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        histogram_naive_kernel<<<grid, block, numBins * sizeof(int)>>>(d_data, d_finalHist, N, numBins);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        double elapsedSec = elapsedTime / 1000.0;
        double opsPerSec = totalOps / elapsedSec;
        double measuredGops = opsPerSec / 1e9;
        
        fprintf(fp, "Naive-%d,%f\n", numBins, measuredGops);
    }
    
    // optimized kernel
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];
        size_t sharedMemSizeOpt = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
        cudaMemset(d_partialHist, 0, gridSize * numBins * sizeof(int));
        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        histogram_optimized_kernel<<<grid, block, sharedMemSizeOpt>>>(d_data, d_partialHist, N, numBins);
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
        
        fprintf(fp, "Optimized-%d,%f\n", numBins, measuredGops);
    }
    
    // tiled kernel
    for (int bi = 0; bi < 7; bi++) {
        int numBins = binSizes[bi];
        size_t sharedMemSizeTile = (numWarps * numBins) * sizeof(int);
        cudaMemset(d_partialHist, 0, gridSize * numBins * sizeof(int));
        cudaMemset(d_finalHist, 0, numBins * sizeof(int));
        
        cudaEventRecord(start, 0);
        histogram_tiled_kernel<<<grid, block, sharedMemSizeTile>>>(d_data, d_partialHist, N, numBins);
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
        
        fprintf(fp, "Tiled-%d,%f\n", numBins, measuredGops);
    }
    
    fclose(fp);
    
    // cleanup
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}