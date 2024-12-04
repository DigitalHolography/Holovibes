#include "otsu.cuh"
#include "common.cuh"
#include "cuComplex.h"
#include "cuda_runtime.h"
#include "hardware_limits.hh"
#include "cuda_memory.cuh"
#include "tools_analysis_debug.hh"

using uint = unsigned int;

__global__ void global_threshold_kernel(float* input, int size, float globalThreshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        input[idx] = input[idx] > globalThreshold;
}

void compute_binarise_otsu(
    float* input_output, float threshold, const size_t width, const size_t height, const cudaStream_t stream)
{
    size_t img_size = width * height;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(img_size, threads);

    global_threshold_kernel<<<blocks, threads, 0, stream>>>(input_output, img_size, threshold);
    cudaXStreamSynchronize(stream);
}
