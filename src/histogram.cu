#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
#include <math.h>
using namespace cooperative_groups;

// Histogram kernel using double buffering, shared memory and cooperative groups.
// Input data values: [0,1023] and numBins must be 2^k where k in [2,8].
__global__ void histogram_optimized_kernel(const int *data, int *partialHist, int N, int numBins) {
    extern __shared__ int sharedMem[];
    // Basic parameters.
    int blockThreads = blockDim.x * blockDim.y;
    const int warpSize = 32;
    int numWarps = blockThreads / warpSize;
    // Each tile holds blockThreads*4 ints (vectorized loads with int4).
    int tileSizeInts = blockThreads * 4;
    
    // Shared memory pointers.
    int* tile0   = sharedMem;                           // first tile buffer
    int* tile1   = sharedMem + tileSizeInts;              // second tile buffer
    int* warpHist = sharedMem + 2 * tileSizeInts;          // per-warp histogram buffer

    // Cooperative groups.
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);

    int tid     = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane    = warp.thread_rank();

    // Initialize warpHist to zero (with padding).
    int totalWarpHistEntries = numWarps * (numBins + 1);
    for (int i = lane; i < totalWarpHistEntries; i += warpSize) {
        if ((i % (numBins+1)) < numBins)
            warpHist[i] = 0;
    }
    block.sync();

    // Precompute bit-shift factor.
    int k = 0, temp = numBins;
    while (temp > 1) { k++; temp >>= 1; }
    int shift = 10 - k;  // e.g., for numBins = 8, shift = 7.

    // (Re)initialize per-warp histogram region.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    block.sync();

    // Compute global tile size (in ints) and first tile offset.
    // Note: each block processes tileSizeInts ints.
    int globalTileSizeInts = gridDim.x * tileSizeInts;
    int firstOffset = blockIdx.x * tileSizeInts;

    // Load first tile into tile0.
    if (firstOffset < N) {
        int globalIndex = firstOffset + tid * 4;
        if (globalIndex + 3 < N) {
            int4 tmp = __ldg(reinterpret_cast<const int4*>(&data[globalIndex]));
            tile0[tid*4+0] = tmp.x;
            tile0[tid*4+1] = tmp.y;
            tile0[tid*4+2] = tmp.z;
            tile0[tid*4+3] = tmp.w;
        } else {
            for (int i = 0; i < 4; i++) {
                int idx = globalIndex + i;
                tile0[tid*4+i] = (idx < N) ? __ldg(&data[idx]) : -1;
            }
        }
    }
    block.sync();

    // Process tiles using double buffering.
    for (int offset = firstOffset + globalTileSizeInts; offset < N; offset += globalTileSizeInts) {
        // Load next tile into tile1.
        int globalIndex = offset + tid * 4;
        bool validLoad = (globalIndex < N);
        if (validLoad) {
            if (globalIndex + 3 < N) {
                int4 tmp = __ldg(reinterpret_cast<const int4*>(&data[globalIndex]));
                tile1[tid*4+0] = tmp.x;
                tile1[tid*4+1] = tmp.y;
                tile1[tid*4+2] = tmp.z;
                tile1[tid*4+3] = tmp.w;
            } else {
                for (int i = 0; i < 4; i++) {
                    int idx = globalIndex + i;
                    tile1[tid*4+i] = (idx < N) ? __ldg(&data[idx]) : -1;
                }
            }
        }

        // Process current tile in tile0.
        {
            // Each thread builds a small local histogram.
            int localHist[8] = {0};
            int localBins[8]  = {-1,-1,-1,-1,-1,-1,-1,-1};

            // Process assigned values.
            for (int i = tid; i < tileSizeInts; i += blockThreads) {
                int value = tile0[i];
                if (value < 0) continue;
                int bin = value >> shift;
                if (bin < 0 || bin >= numBins) continue;
                bool foundBin = false;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (bin == localBins[j]) {
                        localHist[j]++;
                        foundBin = true;
                        break;
                    } else if (localBins[j] == -1) {
                        localBins[j] = bin;
                        localHist[j] = 1;
                        foundBin = true;
                        break;
                    }
                }
                if (!foundBin) {
                    // Flush local histogram to warpHist.
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        if (localHist[j] > 0 && localBins[j] >= 0 && localBins[j] < numBins)
                            atomicAdd(&warpHist[warp_id * numBins + localBins[j]], localHist[j]);
                        localBins[j] = -1;
                        localHist[j] = 0;
                    }
                    localBins[0] = bin;
                    localHist[0] = 1;
                }
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (localHist[j] > 0 && localBins[j] >= 0 && localBins[j] < numBins)
                    atomicAdd(&warpHist[warp_id * numBins + localBins[j]], localHist[j]);
            }
        }
        block.sync();
        // Swap buffers: tile1 becomes current tile.
        int *temp = tile0;
        tile0 = tile1;
        tile1 = temp;
        block.sync();
    }

    // Process final tile in tile0.
    {
        int localHist[8] = {0};
        int localBins[8]  = {-1,-1,-1,-1,-1,-1,-1,-1};
        #pragma unroll
        for (int i = tid; i < tileSizeInts; i += blockThreads) {
            int value = tile0[i];
            if (value < 0) continue;
            int bin = value >> shift;
            if (bin < 0 || bin >= numBins) continue;
            bool foundBin = false;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                if (bin == localBins[j]) {
                    localHist[j]++;
                    foundBin = true;
                    break;
                } else if (localBins[j] == -1) {
                    localBins[j] = bin;
                    localHist[j] = 1;
                    foundBin = true;
                    break;
                }
            }
            if (!foundBin) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (localHist[j] > 0 && localBins[j] >= 0 && localBins[j] < numBins)
                        atomicAdd(&warpHist[warp_id * numBins + localBins[j]], localHist[j]);
                    localBins[j] = -1;
                    localHist[j] = 0;
                }
                localBins[0] = bin;
                localHist[0] = 1;
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (localHist[j] > 0 && localBins[j] >= 0 && localBins[j] < numBins)
                atomicAdd(&warpHist[warp_id * numBins + localBins[j]], localHist[j]);
        }
    }
    block.sync();

    // Reduce per-warp histograms into block-level partial histogram.
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            #pragma unroll
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            // Final reduction using warp shuffles.
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                sum += warp.shfl_down(sum, offset);
            }
            if (lane == 0) {
                partialHist[blockIdx.x * numBins + i] = sum;
            }
        }
    }
}

// Simple reduction kernel that sums block-level partial histograms.
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
 
// Main routine.
int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic -i <BinNum> <VecDim> [GridSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [GridSize]\n", argv[0]);
        return 1;
    }
 
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    // Validate: numBins must be 2^k with k in [2,8].
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k in [2,8].\n");
        return 1;
    }
 
    // Fixed block dimensions: 4x64 threads = 256 threads per block.
    dim3 block(4, 64);
    // Compute tile size in ints: each block processes blockThreads*4 ints.
    int tileSizeInts = block.x * block.y * 4;  // 256*4 = 1024 ints per block.
    int gridSize;
    // If a grid size is provided as argument, use it; otherwise, compute based on tileSizeInts.
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + tileSizeInts - 1) / tileSizeInts;
    dim3 grid(gridSize);
 
    // Calculate shared memory size.
    int numWarps = (block.x * block.y) / 32;
    size_t sharedMemSize = (2 * tileSizeInts + numWarps * numBins) * sizeof(int);
    size_t dataSize = N * sizeof(int);
    size_t partialHistSize = gridSize * numBins * sizeof(int);
    size_t finalHistSize = numBins * sizeof(int);
 
    // Allocate and initialize host memory.
    int *h_data = (int*) malloc(dataSize);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        h_data[i] = rand() % 1024;
 
    // Allocate device memory.
    int *d_data, *d_partialHist, *d_finalHist;
    cudaMalloc((void**)&d_data, dataSize);
    cudaMalloc((void**)&d_partialHist, partialHistSize);
    cudaMalloc((void**)&d_finalHist, finalHistSize);
    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);
    cudaMemset(d_partialHist, 0, partialHistSize);
    cudaMemset(d_finalHist, 0, finalHistSize);
 
    // Create CUDA events for timing.
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
 
    // Print performance info.
    double totalOps = (double) N + (gridSize * numBins);
    double opsPerSec = totalOps / (elapsedTime / 1000.0);
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations: %.0f\n", totalOps);
    printf("Throughput: %e ops/sec\n", opsPerSec);
 
    // (Optional) Print device occupancy.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int coresPerSM = 64;
    int totalCores = deviceProp.multiProcessorCount * coresPerSM;
    double clockHz = deviceProp.clockRate * 1000.0;
    double theoreticalOps = totalCores * clockHz * 2;
    printf("Device: %s\n", deviceProp.name);
    printf("SMs: %d, Cores/SM: %d, Total Cores: %d\n", deviceProp.multiProcessorCount, coresPerSM, totalCores);
    printf("Clock Rate: %0.2f GHz\n", clockHz / 1e9);
    printf("Theoretical Peak Ops/sec (int): %e ops/sec\n", theoreticalOps);
 
    // Copy final histogram from device to host.
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
