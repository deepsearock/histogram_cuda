#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel using per-warp private histograms and register-level accumulation.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Allocate dynamic shared memory for per-warp histograms.
    // Each warp gets numBins integers.
    extern __shared__ int warpHist[]; // size = (numWarps * numBins)
    
    const int warpSize = 32;
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    int numWarps = blockDim.x / warpSize;  // assume blockDim.x is a multiple of 32

    // Each warp initializes its own portion of the shared memory.
    for (int i = lane; i < numBins; i += warpSize) {
        warpHist[warp_id * numBins + i] = 0;
    }
    __syncthreads();  // Ensure all warp histograms are zeroed.

    // Process input data using a strided loop.
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;

    // Use register-level accumulation to batch atomic updates when possible.
    int localBin = -1;
    int localCount = 0;
    while (idx < N) {
        int value = data[idx];
        // Map value in [0, 1023] to a bin [0, numBins-1].
        int bin = value / (1024 / numBins);

        // If the current value maps to the same bin as the previous one,
        // accumulate the count in a register.
        if (bin == localBin) {
            localCount++;
        } else {
            // Flush the previous count.
            if (localCount > 0) {
                atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
            }
            localBin = bin;
            localCount = 1;
        }
        idx += stride;
    }
    // Flush any remaining count.
    if (localCount > 0) {
        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
    }
    __syncthreads();

    // Now, reduce the per-warp histograms into a single block-level histogram.
    // Let the first warp (warp_id == 0) perform the reduction.
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            // Sum the contributions for bin i from all warps.
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            // Write the block's partial histogram for bin i into global memory.
            partialHist[blockIdx.x * numBins + i] = sum;
        }
    }
    // Optionally, you can use __syncwarp() here for finer-grained sync within the first warp.
}

// Reduction kernel: sum the partial histograms from all blocks into the final histogram.
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
    // Command-line usage:
    // ./histogram_atomic -i <BinNum> <VecDim> [BlockSize] [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize] [GridSize]\n", argv[0]);
        return 1;
    }

    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);

    // Validate numBins: must be 2^k with k between 2 and 8 (i.e. 4, 8, 16, 32, 64, 128, or 256).
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g. 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }

    // Optionally accept block and grid sizes.
    int blockSize = 256;
    int gridSize;
    if (argc >= 5)
        blockSize = atoi(argv[4]);
    if (argc >= 6)
        gridSize = atoi(argv[5]);
    else
        gridSize = (N + blockSize - 1) / blockSize;

    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);

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
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);

    // Copy input data to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);

    // Create CUDA events for performance measurement.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Launch the optimized histogram kernel.
    // Allocate shared memory: (numWarps * numBins) integers.
    int numWarps = blockSize / 32;
    size_t sharedMemSize = numWarps * numBins * sizeof(int);
    histogram_optimized_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_data, d_partialHist, N, numBins);

    // Launch the reduction kernel to combine partial histograms.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Measure elapsed time.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Compute a throughput metric (each atomic update or merge write counted as an operation).
    double totalOps = (double) N + (gridSize * numBins); // approximate total operations
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double gflops = opsPerSec / 1e9;  // "TFLOPS" metric

    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
    printf("Performance: %f TFLOPS\n", gflops);

    // Copy final histogram back to host.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);

    // Optionally print nonzero bins.
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Bin %d: %d\n", i, h_finalHist[i]);
    }

    // Clean up.
    free(h_data);
    free(h_finalHist);
    cudaFree(d_data);
    cudaFree(d_partialHist);
    cudaFree(d_finalHist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
