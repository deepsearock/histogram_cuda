#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void hierarchical_histogram_kernel(const int *data, int *partialHist, int N, int numBins) {
    // Shared memory is used for per-warp histograms.
    // Each warp gets its own segment of size "numBins".
    extern __shared__ int warpHist[];

    // Compute thread and warp indices.
    const int blockThreads = blockDim.x * blockDim.y;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warpSize = 32;
    const int warp_id = tid / warpSize;
    const int lane = tid % warpSize;
    const int numWarps = blockThreads / warpSize;

    // Initialize the shared memory for each warp's histogram.
    for (int i = tid; i < numWarps * numBins; i += blockThreads) {
        warpHist[i] = 0;
    }
    __syncthreads();

    // Determine the bit-shift factor.
    // Data values are in [0,1023] and bins are 2^k, so each bin spans 1024/numBins values.
    int shift = 10;
    int tmp = numBins;
    while (tmp > 1) {
        shift--;
        tmp >>= 1;
    }

    // Each thread keeps its own private histogram in registers.
    // We assume numBins is small (<=256) so that an array of that size fits in registers.
    int localHist[256];
    for (int b = 0; b < numBins; b++) {
        localHist[b] = 0;
    }

    // Set the number of integers each thread loads per iteration.
    // Here we choose 8 integers per thread.
    const int intsPerThread = 8;
    // A block processes a tile of data of size:
    const int tileSize = blockThreads * intsPerThread;
    
    // Each block processes a contiguous tile from global memory.
    // Compute the base index for this block.
    int blockBase = blockIdx.x * tileSize;
    // The grid-stride step is tileSize * gridDim.x.
    int stride = tileSize * gridDim.x;

    // Now each thread processes a sequence of intsPerThread-wide chunks.
    // We compute a starting index based on blockBase plus the thread’s relative offset.
    for (int i = blockBase + tid * intsPerThread; i < N; i += stride) {
        // Use vectorized loads if the full 8 integers are in-range.
        if (i + intsPerThread - 1 < N) {
            // Two int4 loads bring 8 integers.
            int4 vec1 = ((const int4*)data)[ (i     ) / 4 ];
            int4 vec2 = ((const int4*)data)[ (i + 4 ) / 4 ];
            int vals[8];
            vals[0] = vec1.x; vals[1] = vec1.y; vals[2] = vec1.z; vals[3] = vec1.w;
            vals[4] = vec2.x; vals[5] = vec2.y; vals[6] = vec2.z; vals[7] = vec2.w;
            #pragma unroll
            for (int j = 0; j < intsPerThread; j++) {
                int value = vals[j];
                int bin = value >> shift;
                localHist[bin]++;
            }
        } else {
            // Fallback: scalar loads for the tail elements.
            for (int j = 0; j < intsPerThread; j++) {
                int idx = i + j;
                if (idx < N) {
                    int value = data[idx];
                    int bin = value >> shift;
                    localHist[bin]++;
                }
            }
        }
    }

    // Now each thread merges its private histogram into its warp's shared memory region.
    for (int b = 0; b < numBins; b++) {
        atomicAdd(&warpHist[warp_id * numBins + b], localHist[b]);
    }
    __syncthreads();

    // Finally, warp 0 of the block reduces the per-warp histograms into a block-level partial histogram.
    if (warp_id == 0) {
        for (int b = lane; b < numBins; b += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + b];
            }
            partialHist[blockIdx.x * numBins + b] = sum;
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
    
    // Set fixed block dimensions, for example 8 x 32 = 256 threads per block.
    dim3 block(8, 32);
    const int blockThreads = block.x * block.y; // 256 threads
    
    // Each thread loads 8 integers per iteration.
    const int intsPerThread = 8;
    // Each block processes a tile of size:
    int tileSize = blockThreads * intsPerThread; // 256 * 8 = 2048 integers per block.
    
    // Compute grid size so that all N integers are processed.
    int gridSize;
    if (argc >= 5)
        gridSize = atoi(argv[4]);
    else
        gridSize = (N + tileSize - 1) / tileSize;
    dim3 grid(gridSize);
    
    // Shared memory: one integer per bin per warp.
    int numWarps = blockThreads / 32; // 256 / 32 = 8.
    size_t sharedMemSize = numWarps * numBins * sizeof(int);
    
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
    
    // Launch the hierarchical histogram kernel.
    hierarchical_histogram_kernel<<<grid, block, sharedMemSize>>>(d_data, d_partialHist, N, numBins);
    
    // Launch the reduction kernel.
    int reduceBlockSize = 256;
    int reduceGridSize = (numBins + reduceBlockSize - 1) / reduceBlockSize;
    histogram_reduce_kernel<<<reduceGridSize, reduceBlockSize>>>(d_partialHist, d_finalHist, numBins, gridSize);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate throughput (approximate atomic operations metric).
    double totalOps = (double) N + (gridSize * numBins);
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total kernel execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // (Optional) Copy final histogram from device to host and print histogram bins.
    int *h_finalHist = (int*) malloc(finalHistSize);
    cudaMemcpy(h_finalHist, d_finalHist, finalHistSize, cudaMemcpyDeviceToHost);
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