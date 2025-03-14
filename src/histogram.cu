#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering (for input tiling),
// vectorized (int4) loads, and per-warp histograms.
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// New optimized histogram kernel using per-thread private histograms and shared memory reduction.
// Assumptions:
//   - The input values are in the range [0, 1023].
//   - numBins is <= 256 (so we can allocate a fixed-size local array of 256 ints).
__global__ void histogram_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Each thread keeps a private histogram in registers.
    int localHist[256];  // Maximum allowed numBins is 256.
#pragma unroll
    for (int i = 0; i < numBins; i++) {
        localHist[i] = 0;
    }

    // Compute global thread index and overall stride.
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // Process data using vectorized loads (int4) for most of the input.
    int numInt4 = N / 4;
    for (int i = globalThreadId; i < numInt4; i += totalThreads) {
        // Vectorized load.
        int4 vec = ((const int4*)data)[i];
        // Process each component.
        int vals[4] = { vec.x, vec.y, vec.z, vec.w };
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int val = vals[j];
            // Map [0,1023] into [0,numBins-1] using bit-shift division by 1024.
            int bin = (val * numBins) >> 10;  // Equivalent to (val * numBins) / 1024.
            localHist[bin]++;
        }
    }
    // Handle any remaining elements (tail case).
    int start = numInt4 * 4;
    for (int i = start + globalThreadId; i < N; i += totalThreads) {
        int val = data[i];
        int bin = (val * numBins) >> 10;
        localHist[bin]++;
    }

    // Allocate shared memory for block-level reduction.
    // Each thread writes its private histogram into a unique slot.
    extern __shared__ int sdata[]; // Size should be blockDim.x * numBins * sizeof(int)
    for (int i = 0; i < numBins; i++) {
        sdata[threadIdx.x * numBins + i] = localHist[i];
    }
    __syncthreads();

    // Reduce the per-thread histograms into a single block histogram.
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            for (int i = 0; i < numBins; i++) {
                sdata[threadIdx.x * numBins + i] += sdata[(threadIdx.x + offset) * numBins + i];
            }
        }
        __syncthreads();
    }

    // Write the block's partial histogram to global memory.
    if (threadIdx.x == 0) {
        for (int i = 0; i < numBins; i++) {
            partialHist[blockIdx.x * numBins + i] = sdata[i];
        }
    }
}

// The reduction kernel (unchanged) sums partial histograms from each block.
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
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [BlockSize] [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize] [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // Optionally accept block and grid sizes.
    int blockSizeTotal = 256; // default total threads per block
    int gridSize;
    if (argc >= 5)
        blockSizeTotal = atoi(argv[4]);
    if (argc >= 6)
        gridSize = atoi(argv[5]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    // Set 2D block shape: force blockDim.x = 32, blockDim.y = blockSizeTotal / 32.
    int blockDimX = 32;
    int blockDimY = blockSizeTotal / 32;
    if (blockDimY < 1) blockDimY = 1;
    dim3 block(blockDimX, blockDimY);
    dim3 grid(gridSize);
    
    // Calculate shared memory size:
    //   Two tile buffers: 2 * tileSizeInts, where tileSizeInts = block.x * block.y * 4.
    //   Plus per-warp histogram: numWarps * numBins integers.
    int tileSizeInts = block.x * block.y * 4;
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    
    // Allocate and initialize host memory.
    int *h_data = (int*) malloc(dataSize);
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
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    
    // Copy input data to device.
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    
    // Create CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    // Launch the optimized histogram kernel.
    histogram_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
    // Launch the reduction kernel.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate measured throughput based on approximate atomic operations.
    double totalOps = (double) N + (gridSize * numBins); // approximate total operations
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // Calculate occupancy.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;  // e.g., 2048 for V100
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_kernel, blockSizeTotal, sharedMemSize);
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;  // percentage
    printf("Occupancy per SM: %f %%\n", occupancy);
    
    // Calculate theoretical peak integer ops (compute) based on GPU specs.
    int coresPerSM = 64; // assumption for Volta-like GPUs
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    double clockHz = deviceProp.clockRate * 1000.0;  // clockRate is in kHz, convert to Hz.
    double theoreticalOpsCompute = totalCores * clockHz * 2;  // 2 ops per cycle
    printf("Theoretical Peak Ops/sec (Compute): %e ops/sec\n", theoreticalOpsCompute);
    
    // Calculate theoretical memory bandwidth (assume double data rate):
    double memClockHz = deviceProp.memoryClockRate * 1000.0;
    double memBusWidthBytes = deviceProp.memoryBusWidth / 8.0;
    // For DDR (or HBM2 double data rate), multiply by 2.
    double theoreticalMemBandwidth = memClockHz * memBusWidthBytes * 2;
    printf("Theoretical Memory Bandwidth: %0.2f GB/s\n", theoreticalMemBandwidth / 1e9);
    
    // Effective operations/sec if memory-bound (each int = 4 bytes).
    double effectiveOpsFromMemory = theoreticalMemBandwidth / 4.0;
    printf("Theoretical Effective Ops/sec (Memory-bound): %e ops/sec\n", effectiveOpsFromMemory);
    
    // (Optional) Copy final histogram from device to host and print nonzero bins.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
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
