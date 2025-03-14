#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// Histogram kernel: each block computes a local histogram in shared memory.
__global__ void histogram_kernel(const int *data, int *hist, int N, int numBins) {
    // Declare dynamic shared memory for the block's histogram.
    extern __shared__ int hist_s[];

    int tid = threadIdx.x;
    // Initialize the shared histogram bins to zero.
    for (int i = tid; i < numBins; i += blockDim.x) {
        hist_s[i] = 0;
    }
    __syncthreads();

    // Each thread processes a strided portion of the input vector.
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    while (idx < N) {
        int value = data[idx];
        // Map the value [0, 1023] to a bin [0, numBins-1].
        int bin = value / (1024 / numBins);
        atomicAdd(&hist_s[bin], 1);
        idx += stride;
    }
    __syncthreads();

    // Merge the block's shared histogram into the global histogram.
    for (int i = tid; i < numBins; i += blockDim.x) {
        atomicAdd(&hist[i], hist_s[i]);
    }
}

int main(int argc, char *argv[]) {
    // Expect command line: ./histogram_atomic -i <BinNum> <VecDim>
    if (argc != 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim>\n", argv[0]);
        return 1;
    }

    // Parse command line parameters.
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Validate numBins: must be 2^k for k from 2 to 8 (i.e. 4, 8, 16, 32, 64, 128, or 256).
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g. 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }

    size_t dataSize = N * sizeof(int);
    size_t histSize = numBins * sizeof(int);

    // Allocate host memory and generate random input data (values between 0 and 1023).
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;
    }

    // Allocate device memory.
    int *d_data, *d_hist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_hist, histSize);

    // Copy the input vector from host to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

    // Initialize the global histogram on the device to zero.
    cudaMemset(d_hist, 0, histSize);

    // Define grid and block dimensions.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing the kernel.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event.
    cudaEventRecord(start, 0);

    // Launch the histogram kernel with dynamic shared memory size (numBins * sizeof(int)).
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, histSize>>>(d_data, d_hist, N, numBins);

    // Record the stop event and wait for the kernel to finish.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute the elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Compute total atomic operations:
    // - One per processed element (from the main loop).
    // - Plus one per bin per block during the merge phase.
    double totalOps = (double) N + (blocksPerGrid * numBins);
    double elapsedSec = elapsedTime / 1000.0;  // Convert ms to seconds.
    double opsPerSec = totalOps / elapsedSec;
    // Report throughput in "TFLOPS" (here, one atomic op is counted as one operation).
    double tflops = opsPerSec / 1e12;

    // Report performance.
    printf("Histogram kernel execution time: %f ms\n", elapsedTime);
    printf("Total atomic operations: %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
    printf("Performance: %f TFLOPS\n", tflops);

    // Copy the computed histogram back to the host.
    int *h_hist = (int*) malloc(histSize);
    cudaMemcpy(h_hist, d_hist, histSize, cudaMemcpyDeviceToHost);

    // Optionally, print out nonzero histogram bins.
    for (int i = 0; i < numBins; i++) {
        if (h_hist[i] != 0)
            printf("Bin %d: %d\n", i, h_hist[i]);
    }

    // Clean up.
    free(h_data);
    free(h_hist);
    cudaFree(d_data);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
