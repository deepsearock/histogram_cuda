#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

// New kernel that computes the histogram for one chunk of data.
// It uses per–warp histograms stored in shared memory and then reduces
// them into a block–level partial histogram.
__global__ void histogram_kernel_chunk(const int *data, int *partialHist, int chunkSize, int numBins) {
    // Use shared memory solely for per–warp histograms.
    extern __shared__ int warpHist[];
    int blockThreads = blockDim.x * blockDim.y;
    const int warpSize = 32;
    int numWarps = blockThreads / warpSize;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;
    
    // Initialize per–warp histograms.
    for (int i = lane; i < numWarps * numBins; i += warpSize) {
        warpHist[i] = 0;
    }
    __syncthreads();
    
    // Process the chunk. Each thread processes multiple elements.
    for (int i = tid; i < chunkSize; i += blockThreads) {
        int value = data[i];
        // Assuming data values are in [0, 1023] and bins evenly divide that range.
        int bin = value / (1024 / numBins);
        atomicAdd(&warpHist[warp_id * numBins + bin], 1);
    }
    __syncthreads();
    
    // Reduce per–warp histograms into a single partial histogram.
    if (warp_id == 0) {
        for (int i = lane; i < numBins; i += warpSize) {
            int sum = 0;
            for (int w = 0; w < numWarps; w++) {
                sum += warpHist[w * numBins + i];
            }
            partialHist[i] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    // Usage: ./histogram_atomic_stream -i <BinNum> <VecDim> [BlockSize]
    if (argc < 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        fprintf(stderr, "Usage: %s -i <BinNum> <VecDim> [BlockSize]\n", argv[0]);
        return 1;
    }
    
    int numBins = atoi(argv[2]);
    int N = atoi(argv[3]);
    
    // Validate numBins: must be 2^k with k between 2 and 8.
    if (numBins < 4 || numBins > 256 || (numBins & (numBins - 1)) != 0) {
        fprintf(stderr, "Error: <BinNum> must be 2^k with k from 2 to 8 (e.g., 4, 8, 16, 32, 64, 128, or 256).\n");
        return 1;
    }
    
    int blockSizeTotal = 256; // default threads per block
    if (argc >= 5)
        blockSizeTotal = atoi(argv[4]);
    
    // Set block dimensions: force blockDim.x = 32, blockDim.y = blockSizeTotal/32.
    int blockDimX = 32;
    int blockDimY = blockSizeTotal / 32;
    if (blockDimY < 1) blockDimY = 1;
    dim3 block(blockDimX, blockDimY);
    // In this design each kernel launch uses a single block.
    int gridSize = 1;
    
    // Allocate and initialize host input data.
    int *h_data = (int*) malloc(N * sizeof(int));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory for input data.\n");
        return 1;
    }
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % 1024;  // values in [0, 1023]
    }
    
    // Register host memory to allow asynchronous copies.
    cudaHostRegister(h_data, N * sizeof(int), cudaHostRegisterDefault);
    
    // Define a chunk size (in ints) for processing.
    // For example, process 1M ints per kernel launch (or less if N is smaller).
    int chunkSize = (1 << 20);
    if (chunkSize > N)
        chunkSize = N;
    int numChunks = (N + chunkSize - 1) / chunkSize;
    
    // Allocate two device buffers for input data (for double–streaming).
    int *d_data[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc((void**)&d_data[i], chunkSize * sizeof(int));
    }
    
    // Allocate two device buffers for partial histograms.
    int *d_partialHist[2];
    for (int i = 0; i < 2; i++) {
        cudaMalloc((void**)&d_partialHist[i], numBins * sizeof(int));
    }
    
    // Allocate host memory for the final histogram and initialize to zero.
    int *h_finalHist = (int*) calloc(numBins, sizeof(int));
    
    // Create two CUDA streams.
    cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Determine shared memory size for the kernel.
    int blockThreads = block.x * block.y;
    const int warpSize = 32;
    int numWarps = blockThreads / warpSize;
    size_t sharedMemSize = numWarps * numBins * sizeof(int);
    
    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // Process each chunk using a ping–pong scheme across the two streams.
    for (int chunk = 0; chunk < numChunks; chunk++) {
        int streamId = chunk % 2;
        // Calculate the number of ints to process in this chunk.
        int currentChunkSize = (chunk < numChunks - 1) ? chunkSize : (N - chunk * chunkSize);
        
        // Asynchronously copy the current chunk from host to device.
        cudaMemcpyAsync(d_data[streamId],
                        h_data + chunk * chunkSize,
                        currentChunkSize * sizeof(int),
                        cudaMemcpyHostToDevice,
                        streams[streamId]);
        
        // Zero out the device partial histogram buffer.
        cudaMemsetAsync(d_partialHist[streamId], 0, numBins * sizeof(int), streams[streamId]);
        
        // Launch the histogram kernel for the current chunk.
        histogram_kernel_chunk<<<gridSize, block, sharedMemSize, streams[streamId]>>>
            (d_data[streamId], d_partialHist[streamId], currentChunkSize, numBins);
        
        // Allocate a temporary host buffer to hold the partial histogram.
        int *h_partialHist = (int*) malloc(numBins * sizeof(int));
        // Asynchronously copy the partial histogram from device to host.
        cudaMemcpyAsync(h_partialHist,
                        d_partialHist[streamId],
                        numBins * sizeof(int),
                        cudaMemcpyDeviceToHost,
                        streams[streamId]);
        
        // Wait for the stream to finish processing this chunk.
        cudaStreamSynchronize(streams[streamId]);
        
        // Accumulate the partial histogram into the final histogram.
        for (int i = 0; i < numBins; i++) {
            h_finalHist[i] += h_partialHist[i];
        }
        free(h_partialHist);
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Calculate performance metrics (approximate, using number of ints processed).
    double totalOps = (double) N;
    double elapsedSec = elapsedTime / 1000.0;
    double opsPerSec = totalOps / elapsedSec;
    double measuredGFlops = opsPerSec / 1e9;
    
    printf("Total execution time: %f ms\n", elapsedTime);
    printf("Total operations (approx.): %.0f\n", totalOps);
    printf("Measured Throughput: %e ops/sec\n", opsPerSec);
    printf("Measured Performance: %f GFLOPS (atomic ops metric)\n", measuredGFlops);
    
    // Print the nonzero bins of the final histogram.
    for (int i = 0; i < numBins; i++) {
        if (h_finalHist[i] != 0)
            printf("Bin %d: %d\n", i, h_finalHist[i]);
    }
    
    // Clean up device memory and CUDA streams.
    for (int i = 0; i < 2; i++) {
        cudaFree(d_data[i]);
        cudaFree(d_partialHist[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Unregister and free host memory.
    cudaHostUnregister(h_data);
    free(h_data);
    free(h_finalHist);
    
    return 0;
}
