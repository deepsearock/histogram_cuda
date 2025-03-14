#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Optimized histogram kernel with double buffering, vectorized loads,
// per-warp histograms, bit-shift based bin calculation, and loop unrolling.
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory layout: [tile0 | tile1 | per-warp histograms]
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    const int blockThreads = blockDim.x * blockDim.y;
    // Each thread loads 32 integers.
    const int intsPerThread = 16;
    const int tileSizeInts = blockThreads * intsPerThread; // total integers per tile

    // Pointers to two tile buffers for double buffering.
    int *tile0 = sharedMem;
    int *tile1 = sharedMem + tileSizeInts;
    // Per-warp histogram region follows the two tiles.
    const int numWarps = blockThreads / warpSize;
    int *warpHist = sharedMem + 2 * tileSizeInts; // length: numWarps * numBins

    // Precompute the bit-shift factor (assumes numBins is 2^k, with k between 2 and 8).
    int k = 0;
    int tmp = numBins;
    while (tmp > 1) {
        k++;
        tmp >>= 1;
    }
    const int shift = 10 - k;  // e.g., if numBins==8 then shift = 7

    // Flatten thread index.
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;

    // Initialize per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Each block processes a contiguous segment of global data.
    // Compute the base offset for this block and the stride for the grid-stride loop.
    const int baseOffset = blockIdx.x * tileSizeInts;
    const int stride = gridDim.x * tileSizeInts;
    int offset = baseOffset;

    // === Load the first tile into tile0 ===
    if (offset < N) {
        int globalIndex = offset + tid * intsPerThread;
        if (globalIndex + intsPerThread <= N) {
            // Vectorized loads via eight int4 loads.
            #pragma unroll
            for (int i = 0; i < intsPerThread / 4; i++) {
                int4 tmp = ((const int4*)data)[ (globalIndex + i * 4) / 4 ];
                int base = tid * intsPerThread + i * 4;
                tile0[base + 0] = tmp.x;
                tile0[base + 1] = tmp.y;
                tile0[base + 2] = tmp.z;
                tile0[base + 3] = tmp.w;
            }
        } else {
            // Fallback: load one integer at a time.
            for (int i = 0; i < intsPerThread; i++) {
                int idx = globalIndex + i;
                tile0[tid * intsPerThread + i] = (idx < N) ? data[idx] : -1;
            }
        }
    }
    __syncthreads();

    // === Double-buffered processing loop ===
    // In each iteration, we load the next tile into tile1 (if available),
    // process the current tile in tile0, then swap buffers.
    while (true) {
        int nextOffset = offset + stride;
        bool hasNextTile = (nextOffset < N);
        if (hasNextTile) {
            int globalIndex = nextOffset + tid * intsPerThread;
            if (globalIndex + intsPerThread <= N) {
                #pragma unroll
                for (int i = 0; i < intsPerThread / 4; i++) {
                    int4 tmp = ((const int4*)data)[ (globalIndex + i * 4) / 4 ];
                    int base = tid * intsPerThread + i * 4;
                    tile1[base + 0] = tmp.x;
                    tile1[base + 1] = tmp.y;
                    tile1[base + 2] = tmp.z;
                    tile1[base + 3] = tmp.w;
                }
            } else {
                for (int i = 0; i < intsPerThread; i++) {
                    int idx = globalIndex + i;
                    tile1[tid * intsPerThread + i] = (idx < N) ? data[idx] : -1;
                }
            }
        }
        __syncthreads();

        // === Process the current tile in tile0 ===
        {
            // Each thread processes its own contiguous segment.
            int start = tid * intsPerThread;
            int end   = start + intsPerThread;
            int localBin = -1;
            int localCount = 0;
            for (int i = start; i < end; i++) {
                int value = tile0[i];
                if (value < 0) continue;
                int bin = value >> shift;
                if (bin == localBin) {
                    localCount++;
                } else {
                    if (localCount > 0)
                        atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
                    localBin = bin;
                    localCount = 1;
                }
            }
            if (localCount > 0)
                atomicAdd(&warpHist[warp_id * numBins + localBin], localCount);
        }
        __syncthreads();

        // If there is no next tile, we are done.
        if (!hasNextTile)
            break;

        // Swap tile buffers so that tile1 (newly loaded) becomes the current tile.
        int *temp = tile0;
        tile0 = tile1;
        tile1 = temp;
        offset += stride;
        __syncthreads();
    }
    
    __syncthreads();

    // === Reduce per-warp histograms into a block-level (partial) histogram ===
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            partialHist[blockIdx.x * numBins + i] = sum;
        }
    }
}




// Reduction kernel: Sum partial histograms from all blocks into the final histogram.
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
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    // (Note: We now use a reduced block size to keep shared memory usage within device limits.)
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    // Use a reduced block size (4 x 32 = 128 threads) to lower shared memory usage.
    dim3 block(4, 64);
    const int blockSizeTotal = block.x * block.y; // 128 threads per block.
    
    // Each thread now loads 32 integers.
    int intsPerThread = 16;
    int tileSizeInts = blockSizeTotal * intsPerThread;  // 128 * 32 = 4096 integers per block.
    
    // Compute grid size based on the new tile size.
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + tileSizeInts - 1) / tileSizeInts;
    dim3 grid(gridSize);
    
    // Calculate shared memory size:
    // Two tile buffers (each of tileSizeInts integers) plus per-warp histograms.
    int numWarps = blockSizeTotal / 32; // 128/32 = 4.
    size_t sharedMemSize = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    printf("Shared memory requested: %zu bytes\n", sharedMemSize);
    
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
    histogram_optimized_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Launch the reduction kernel.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Reduction kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
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
    int maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
    int activeBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks, histogram_optimized_kernel, blockSizeTotal, sharedMemSize);
    float occupancy = (activeBlocks * blockSizeTotal) / (float) maxThreadsPerSM;
    occupancy = occupancy * 100.0f;
    printf("Occupancy per SM: %f %%\n", occupancy);
    
    // Display device properties.
    int coresPerSM = 64; // (adjust if needed)
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    double clockHz = deviceProp.clockRate * 1000.0;
    double theoreticalOps = totalCores * clockHz * 2;
    printf("Device: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    printf("Cores per SM: %d\n", coresPerSM);
    printf("Total CUDA Cores: %d\n", totalCores);
    printf("Clock Rate: %0.2f GHz\n", clockHz / 1e9);
    printf("Theoretical Peak Ops/sec (int): %e ops/sec\n", theoreticalOps);
    
    // Copy final histogram from device to host.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
    
    // Print all histogram bins.
    printf("Final Histogram Bins:\n");
    for (int i = 0; i < numBins; i++) {
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