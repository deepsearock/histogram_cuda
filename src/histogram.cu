#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// First kernel: compute per-block (partial) histograms using shared memory.
__global__ void histogram_partial_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Allocate dynamic shared memory for this block's histogram.
    extern __shared__ int localHist[];
    
    int tid = threadIdx.x;
    // Initialize the local (shared) histogram.
    for (int i = tid; i < numBins; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Each thread processes a strided portion of the input.
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    while (idx < N) {
        int value = data[idx];
        // Map value [0, 1023] to one of the bins.
        int bin = value / (1024 / numBins);
        atomicAdd(&localHist[bin], 1);
        idx += stride;
    }
    __syncthreads();

    // Write the block's partial histogram to global memory.
    for (int i = tid; i < numBins; i += blockDim.x) {
        partialHist[blockIdx.x * numBins + i] = localHist[i];
    }
}

// Second kernel: reduce the partial histograms into a final histogram.
__global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks) {
    int bin = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin < numBins) {
        int sum = 0;
        for (int b = 0; b < numBlocks; b++) {
            sum += partialHist[b * numBins + bin];
        }
        finalHist[bin] = sum;
    }
}

int main(int argc, char *argv[]) {
    // Expected command-line usage:
    // ./histogram_atomic -i <BinNum> <VecDim> [BlockSize] [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize] [GridSize]\n", argv[0]);
        return 1;
    }

    // Parse input parameters.
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Validate numBins: must be a power of 2 with k from 2 to 8 (i.e. 4 to 256).
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g. 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }

    // Optionally accept custom block and grid sizes.
    int blockSize = 256;
    int gridSize;
    if (argc >= 5)
        blockSize = atoi(argv[4]);
    if (argc >= 6)
        gridSize = atoi(argv[5]);
    else
        gridSize = (N + blockSize - 1) / blockSize;

    size_t dataSize = N * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);

    // Allocate and initialize host memory for the input vector.
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input vector.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;  // Random integers in [0, 1023]
    }

    // Allocate device memory.
    int *d_data, *d_finalHist, *d_partialHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);

    // Copy input data from host to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_finalHist, 0, finalHistSize);
    cudaMemset(d_partialHist, 0, partialHistSize);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event.
    cudaEventRecord(start, 0);

    // Launch the partial histogram kernel.
    // Shared memory size is set to numBins * sizeof(int).
    histogram_partial_kernel<<<gridSize, blockSize, numBins * sizeof(int)>>>(d_data, d_partialHist, N, numBins);

    // Launch the reduction kernel to combine the partial histograms.
    // We launch one thread per bin.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);

    // Record stop event and synchronize.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Compute total operations:
    // - In the partial histogram kernel, one atomic op per processed element.
    // - Plus gridSize * numBins writes during the local-to-global merge.
    double totalOps = (double)N + (double)(gridSize * numBins);
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double tflops = opsPerSec / 1e12;  // Note: Here "TFLOPS" is a throughput metric.

    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (atomic updates + merge writes): %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
    printf("Performance: %f TFLOPS\n", tflops);

    // Copy final histogram from device to host.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);

    // Optionally, print nonzero bins.
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Bin %d: %d\n", i, h_finalHist[i]);
    }

    // Clean up.
    free(h_data);
    free(h_finalHist);
    cudaFree(d_data);
    cudaFree(d_finalHist);
    cudaFree(d_partialHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
