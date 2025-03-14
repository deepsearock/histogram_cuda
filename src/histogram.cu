#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Highly optimized histogram kernel that focuses on memory access patterns
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Use cooperative groups for warp-level operations
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int warpSize = 32;
    const int laneMask = warpSize - 1;
    const int laneId = threadIdx.x & laneMask;
    const int warpId = threadIdx.x / warpSize;
    const int blockThreads = blockDim.x;
    const int warpsPerBlock = blockThreads / warpSize;
    
    // Adjusted shared memory to reduce bank conflicts
    // Add padding between warps to avoid bank conflicts (32 banks on modern GPUs)
    extern __shared__ int sharedMem[];
    
    // Calculate bin mask (assuming numBins is power of 2)
    const int binMask = numBins - 1;
    const int binShift = 10 - __popc(binMask); // log2(1024/numBins)
    
    // Register-based local histograms - one per thread
    // Limit to 32 bins in registers, use loop for larger bins
    int localHist[32] = {0}; // Adjust size based on max expected numBins
    
    // Each block processes multiple chunks of data using persistent threads pattern
    const int threadsPerGrid = blockDim.x * gridDim.x;
    const int elementsPerThread = 16; // Process more elements per thread for better ILP
    
    // Process data in chunks directly from global memory
    for (int base = blockIdx.x * blockDim.x + threadIdx.x; 
         base < N; 
         base += threadsPerGrid) {
        
        // Each thread processes multiple elements to improve instruction-level parallelism
        for (int offset = 0; offset < elementsPerThread; offset++) {
            int idx = base + offset * threadsPerGrid;
            if (idx < N) {
                int value = data[idx];
                // Avoid branch divergence by using predication
                int bin = (value >> binShift) & binMask;
                // Increment local histogram counter
                localHist[bin] += (idx < N) ? 1 : 0;
            }
        }
    }
    
    // Use shared memory for per-warp histograms
    int *warpHist = sharedMem + warpId * numBins * 2; // Double padding to avoid conflicts
    
    // Initialize warp histograms - each lane initializes one element
    for (int bin = laneId; bin < numBins; bin += warpSize) {
        warpHist[bin * 2] = 0;
    }
    block.sync();
    
    // Reduce thread-local histograms to warp-level histograms using warp shuffles
    for (int bin = 0; bin < numBins; bin++) {
        // Warp-level reduction using shuffle operations
        int warpSum = cg::reduce(warp, localHist[bin], cg::plus<int>());
        
        // Only lane 0 updates the warp histogram (no atomics within a warp)
        if (laneId == 0 && warpSum > 0) {
            warpHist[bin * 2] = warpSum;
        }
    }
    block.sync();
    
    // Final reduction: first warp reduces all the warp histograms to block histogram
    if (warpId == 0) {
        for (int bin = laneId; bin < numBins; bin += warpSize) {
            int sum = 0;
            for (int w = 0; w < warpsPerBlock; w++) {
                sum += sharedMem[w * numBins * 2 + bin * 2];
            }
            // Write directly to global memory - one atomic per block per bin
            if (sum > 0) {
                atomicAdd(&partialHist[blockIdx.x * numBins + bin], sum);
            }
        }
    }
}

// Optimized reduction kernel that utilizes coalesced memory access
__global__ void histogram_reduce_kernel(const int *partialHist, int *finalHist, int numBins, int numBlocks) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int lane = threadIdx.x & 31;
    
    if (tid < numBins) {
        // Each thread handles one bin across all blocks
        int sum = 0;
        
        // Coalesced memory access pattern - consecutive threads read consecutive memory
        for (int b = 0; b < numBlocks; b++) {
            sum += partialHist[b * numBins + tid];
        }
        
        // Write result to final histogram
        finalHist[tid] = sum;
    }
}

int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // Optimize block size - use 256 threads (8 warps) for better occupancy
    const int blockSize = 256;
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + blockSize * 16 - 1) / (blockSize * 16); // Each thread processes 16 elements
    
    // Calculate shared memory size: per-warp histograms with padding to avoid bank conflicts
    size_t sharedMemSize = (blockSize / 32) * numBins * 2 * sizeof(int);
    
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
    
    // Allocate and initialize host memory
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;  // values in [0, 1023]
    }
    
    // Allocate device memory
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    
    // Copy input data to device
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
    
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);
    
    // Launch optimized histogram kernel
    histogram_optimized_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
    // Launch optimized reduction kernel
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate measured throughput based on approximate atomic operations
    double totalOps = (double) N + (gridSize * numBins); // approximate total operations
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // Calculate occupancy
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSize, sharedMemSize);
    float occupancy = (activeBlocks * blockSize) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;
    printf("Occupancy per SM: %f %%\n", occupancy);
    
    // Display device properties
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
