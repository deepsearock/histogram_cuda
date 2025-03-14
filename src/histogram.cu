#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// Optimized histogram kernel with asynchronous shared memory copies and
// warp-level reduction.
////////////////////////////////////////////////////////////////////////////////
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory layout: [tile0 | tile1 | per-warp histograms]
    extern __shared__ int sharedMem[];

    const int warpSize = 32;
    int blockThreads = blockDim.x * blockDim.y;
    // Each tile holds blockThreads * 4 ints (using vectorized loads of int4)
    int tileSizeInts = blockThreads * 4;
    int *tile0 = sharedMem;                         // first tile buffer
    int *tile1 = sharedMem + tileSizeInts;            // second tile buffer
    int numWarps = blockThreads / warpSize;           // assume blockThreads is a multiple of 32
    int *warpHist = (int*)(sharedMem + 2 * tileSizeInts); // per-warp histogram region

    // Precompute bit-shift factor. (1024 is the maximum value.)
    int k = 0, temp = numBins;
    while (temp > 1) { k++; temp >>= 1; }
    int shift = 10 - k;  // equivalent to dividing by (1024/numBins)

    // Flatten the thread index.
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Initialize per-warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Compute the global tile size (in ints) for double buffering.
    int globalTileSizeInts = gridDim.x * tileSizeInts;
    int firstOffset = blockIdx.x * tileSizeInts;

    // --- Asynchronous copy for first tile into tile0 ---
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;
        if (globalIndex + 3 < N) {
            // Copy 16 bytes (4 ints) asynchronously into shared memory.
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :
                : "r"(tile0 + tid*4), "r"(data + globalIndex), "n"(16)
            );
        } else {
            // Fallback: use regular loads for tail elements.
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile0[tid*4 + i] = (idx < N) ? data[idx] : -1;
            }
        }
    }
    // Commit and wait for the asynchronous copy.
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // --- Process tiles in a double-buffered pipelined loop ---
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        // Launch asynchronous copy for next tile into tile1.
        int globalIndex = offset + tid * 4;
        if (globalIndex + 3 < N) {
            asm volatile(
                "cp.async.cg.shared.global [%0], [%1], %2;\n"
                :
                : "r"(tile1 + tid*4), "r"(data + globalIndex), "n"(16)
            );
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile1[tid*4 + i] = (idx < N) ? data[idx] : -1;
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Process the current tile (in tile0) with run-length aggregation.
        {
            int localBin = -1, localCount = 0;
            #pragma unroll
            for (int i = tid; i < tileSizeInts; i += blockThreads) {
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

        // Swap the tile buffers.
        int *tempPtr = tile0;
        tile0 = tile1;
        tile1 = tempPtr;
        __syncthreads();
    }

    // Process the final tile loaded in tile0.
    {
        int localBin = -1, localCount = 0;
        #pragma unroll
        for (int i = tid; i < tileSizeInts; i += blockThreads) {
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

    // --- Final reduction: combine per-warp histograms into a block-level histogram ---
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            // Use warp shuffle reduction to sum across warp lanes.
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }
            if (lane == 0)
                partialHist[blockIdx.x * numBins + i] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Reduction kernel: Sum partial histograms from all blocks into the final histogram.
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// Host code.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    // With fixed block dimensions (32x32) → 1024 threads per block.
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
    
    // Fixed block dimensions: 32 x 32 → 1024 threads per block.
    const int blockSizeTotal = 32 * 32;
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + blockSizeTotal - 1) / blockSizeTotal;
    
    dim3 block(32, 32);
    dim3 grid(gridSize);
    
    // Calculate shared memory size:
    // Two tile buffers: 2 * tileSizeInts, where tileSizeInts = block.x * block.y * 4.
    // Plus per-warp histogram: numWarps * numBins integers.
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
    histogram_optimized_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
    // Launch the reduction kernel.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate measured throughput based on approximate atomic operations.
    double totalOps = (double) N + (gridSize * numBins);
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // (Optional) Retrieve final histogram and print nonzero bins.
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
