#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define NUM_BINS 1024

// CUDA kernel: Each block builds a local histogram in shared memory.
__global__ void histogram_kernel(const int *data, int *hist, int N) {
    // Allocate shared memory for the block's histogram.
    __shared__ int hist_s[NUM_BINS];

    int tid = threadIdx.x;
    // Each thread initializes part of the shared histogram.
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        hist_s[i] = 0;
    }
    __syncthreads();

    // Process the input vector in a strided manner.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < N) {
        int val = data[idx];
        // Atomically update the shared memory histogram.
        atomicAdd(&hist_s[val], 1);
        idx += stride;
    }
    __syncthreads();

    // Merge the block's shared histogram into the global histogram.
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&hist[i], hist_s[i]);
    }
}

int main(int argc, char* argv[]) {
    // Allow arbitrary input size; default to 1 million elements.
    int N = 1 << 20;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    size_t dataSize = N * sizeof(int);

    // Allocate host memory and generate random input data in the range [0, NUM_BINS-1].
    int *h_data = (int*) malloc(dataSize);
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % NUM_BINS;
    }

    // Allocate device memory.
    int *d_data, *d_hist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_hist, NUM_BINS * sizeof(int));

    // Copy the input data from host to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

    // Initialize the global histogram on the device to zero.
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));

    // Define grid and block dimensions.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing the kernel.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Launch the histogram kernel.
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, N);

    // Record the stop event and synchronize.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Calculate the total number of operations.
    // For this example, we treat each element processed as one operation.
    double totalOps = (double) N;  // One atomic update per data element.
    double elapsedSec = elapsedTime / 1000.0;  // Convert ms to seconds.
    double opsPerSec = totalOps / elapsedSec;
    
    // Compute "TFLOPS" as (ops per second / 1e12). 
    // (Remember: here one atomic op is counted as one flop.)
    double tflops = opsPerSec / 1e12;
    
    // Print performance results.
    printf("Histogram kernel execution time: %f ms\n", elapsedTime);
    printf("Total atomic operations: %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
    printf("Performance: %f TFLOPS\n", tflops);

    // Copy the histogram result back to host memory.
    int *h_hist = (int*) malloc(NUM_BINS * sizeof(int));
    cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Optionally, print non-zero histogram bins.
    for (int i = 0; i < NUM_BINS; i++) {
        if (h_hist[i] != 0)
            printf("Bin %d: %d\n", i, h_hist[i]);
    }

    // Clean up device and host memory, and destroy CUDA events.
    free(h_data);
    free(h_hist);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
