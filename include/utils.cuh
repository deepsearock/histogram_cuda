#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// Structure to hold performance metrics.
struct PerfMetrics {
    double ops;         // Total operations performed.
    float ms;           // Elapsed time in milliseconds.
    double opsPerSec;   // Operations per second.
    double Gops;        // Measured GFLOPS (using atomic ops metric).
};

// Templated function to measure kernel performance.
// Parameters:
//   grid, block, sharedMem - kernel launch configuration,
//   totalOps - number of operations performed by the kernel (calculated externally)
//   kernel - kernel function to launch,
//   args... - kernel arguments.

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

template<typename Kernel, typename... Args>
PerfMetrics measureKernelPerformance(dim3 grid, dim3 block, size_t sharedMem, double totalOps, Kernel kernel, Args... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    kernel<<<grid, block, sharedMem>>>(args...);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    double elapsedSec = elapsedMs / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGops = opsPerSec / 1e9;
    
    PerfMetrics result;
    result.ops = totalOps;
    result.ms = elapsedMs;
    result.opsPerSec = opsPerSec;
    result.Gops = measuredGops;
    
    return result;
}

// Function to allocate device memory for data, partial histogram, and final histogram.
__host__ inline void allocateDeviceMemory(int **d_data, int **d_partialHist, int **d_finalHist,
                                            size_t dataSize, size_t partialHistSize, size_t finalHistSize) {
    cudaMalloc((void**)d_data, dataSize);
    cudaMalloc((void**)d_partialHist, partialHistSize);
    cudaMalloc((void**)d_finalHist, finalHistSize);
}

// Function to copy host input data to device memory and initialize partial & final histograms.
__host__ inline void copyAndInitializeDeviceMemory(int *d_data, const int *h_data, size_t dataSize,
                                                    int *d_partialHist, size_t partialHistSize,
                                                    int *d_finalHist, size_t finalHistSize) {
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
}

#endif // UTILS_CUH