#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "cnn_naive.cuh"  // Include the header file with the naive kernel

// Function to launch the convolution kernel
void cnn_naive(float *h_input, float *h_output, float *h_mask,
               int dimX, int dimY, int dimK) {
    float *d_input, *d_output, *d_mask;
    size_t img_size = dimX * dimY * sizeof(float);
    size_t mask_size = dimK * dimK * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, img_size);
    cudaMalloc((void**)&d_output, img_size);
    cudaMalloc((void**)&d_mask, mask_size);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);
    
    // Set grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((dimX + blockDim.x - 1) / blockDim.x, 
                 (dimY + blockDim.y - 1) / blockDim.y);
    
    // Use the performance measurement function
    float gflops = measurePerformance(d_input, d_mask, d_output, dimX, dimY, dimK, gridDim, blockDim);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
