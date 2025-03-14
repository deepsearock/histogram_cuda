// histogram_atomic.cu
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

namespace cg = cooperative_groups;

// Error-checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if(err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while(0)

// Constants: Adjust blockDim and tile size as needed.
#define BLOCK_SIZE 256
// TILE_SIZE defines how many elements each block processes per iteration.
// Here we use double buffering so TILE_SIZE can be e.g. 2 * BLOCK_SIZE.
#define TILE_SIZE (BLOCK_SIZE * 2)

// Kernel: Each block builds a shared-memory histogram using double buffering
// and then atomically adds its result to the global histogram.
__global__ void histogram_kernel(const int *input, int *global_hist,
                                 int num_elements, int num_bins)
{
    // Cooperative groups for warp tiling.
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    // Shared memory layout:
    // [0, num_bins)         -> shared histogram bins.
    // [num_bins, num_bins + TILE_SIZE)  -> buffer A for input tile.
    // [num_bins + TILE_SIZE, num_bins + 2*TILE_SIZE)  -> buffer B for input tile.
    extern __shared__ int shmem[];
    int *sh_hist = shmem; 
    int *bufferA = shmem + num_bins;
    int *bufferB = bufferA + TILE_SIZE;

    // Initialize shared histogram bins to 0.
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        sh_hist[i] = 0;
    }
    __syncthreads();

    // Setup double buffering: current buffer pointer and next buffer pointer.
    int *currBuffer = bufferA;
    int *nextBuffer = bufferB;

    // Global offset for this block’s processing.
    // Each block processes TILE_SIZE elements per iteration.
    int base_offset = blockIdx.x * TILE_SIZE;
    int elements_in_tile = 0;

    // Load first tile into currBuffer.
    if (base_offset < num_elements) {
        elements_in_tile = (num_elements - base_offset < TILE_SIZE) ?
                           (num_elements - base_offset) : TILE_SIZE;
        // Each thread loads as many as necessary.
        for (int i = threadIdx.x; i < elements_in_tile; i += blockDim.x) {
            currBuffer[i] = input[base_offset + i];
        }
    }
    __syncthreads();

    // Process tiles until all input elements are handled.
    while (base_offset < num_elements) {
        // Preload next tile into nextBuffer if available.
        int next_base = base_offset + TILE_SIZE;
        int next_elements = 0;
        bool hasNextTile = (next_base < num_elements);
        if (hasNextTile) {
            next_elements = ((num_elements - next_base) < TILE_SIZE) ?
                             (num_elements - next_base) : TILE_SIZE;
            for (int i = threadIdx.x; i < next_elements; i += blockDim.x) {
                nextBuffer[i] = input[next_base + i];
            }
        }
        __syncthreads();

        // Process the current tile in currBuffer.
        // Each thread processes elements in a strided loop.
        for (int i = threadIdx.x; i < elements_in_tile; i += blockDim.x) {
            int val = currBuffer[i];
            // Map value (0-1023) to a bin index.
            // For example, if num_bins==16 then each bin covers 1024/16 = 64 values.
            int bin = val / (1024 / num_bins);
            // Use atomic add in shared memory.
            atomicAdd(&sh_hist[bin], 1);
        }
        __syncthreads();

        // Double buffering swap: move nextBuffer to current.
        currBuffer = nextBuffer;
        elements_in_tile = next_elements;
        base_offset += TILE_SIZE;
        // Synchronize before processing next tile.
        __syncthreads();
    }

    // Finally, each block atomically adds its shared histogram to the global histogram.
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&global_hist[i], sh_hist[i]);
    }
}

// Host function to run the histogram.
int main(int argc, char *argv[])
{
    if(argc != 4 || (argv[1][0] != '-' || argv[1][1] != 'i')) {
        printf("Usage: %s -i <BinNum> <VecDim>\n", argv[0]);
        return EXIT_FAILURE;
    }
    int num_bins = atoi(argv[2]);
    int vec_dim = atoi(argv[3]);

    // Validate that num_bins is a power of 2 between 2 and 256.
    bool valid_bins = false;
    for (int k = 2; k <= 8; k++) {
        if(num_bins == (1 << k)) {
            valid_bins = true;
            break;
        }
    }
    if(!valid_bins) {
        fprintf(stderr, "Error: BinNum must be 2^(k) with k between 2 and 8.\n");
        return EXIT_FAILURE;
    }

    // Allocate and initialize host input vector.
    int *h_input = (int *)malloc(vec_dim * sizeof(int));
    if(!h_input) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }
    srand(time(NULL));
    for (int i = 0; i < vec_dim; i++) {
        // Random integers between 0 and 1023.
        h_input[i] = rand() % 1024;
    }

    // Allocate host histogram output.
    int *h_hist = (int *)malloc(num_bins * sizeof(int));
    if(!h_hist) {
        fprintf(stderr, "Host histogram allocation failed\n");
        free(h_input);
        return EXIT_FAILURE;
    }
    // Initialize histogram to zero.
    for (int i = 0; i < num_bins; i++)
        h_hist[i] = 0;

    // Allocate device memory.
    int *d_input = nullptr;
    int *d_hist = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, vec_dim * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_hist, num_bins * sizeof(int)));

    // Copy host input vector to device.
    CUDA_CHECK(cudaMemcpy(d_input, h_input, vec_dim * sizeof(int), cudaMemcpyHostToDevice));
    // Initialize device histogram to zero.
    CUDA_CHECK(cudaMemset(d_hist, 0, num_bins * sizeof(int)));

    // Determine grid dimensions.
    // Each block processes TILE_SIZE elements.
    int grid_size = (vec_dim + TILE_SIZE - 1) / TILE_SIZE;

    // Calculate shared memory size:
    // shared histogram: num_bins integers +
    // two input buffers: 2*TILE_SIZE integers.
    size_t sharedMemSize = num_bins * sizeof(int) + 2 * TILE_SIZE * sizeof(int);

    // Launch the kernel.
    histogram_kernel<<<grid_size, BLOCK_SIZE, sharedMemSize>>>(d_input, d_hist,
                                                                 vec_dim, num_bins);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results from device to host.
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost));

    // Optionally print the histogram.
    printf("Histogram with %d bins:\n", num_bins);
    for (int i = 0; i < num_bins; i++) {
        printf("Bin %d: %d\n", i, h_hist[i]);
    }

    // Cleanup.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_hist));
    free(h_input);
    free(h_hist);

    return EXIT_SUCCESS;
}
