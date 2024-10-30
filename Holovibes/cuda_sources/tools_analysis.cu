#include "cuda_memory.cuh"
#include "common.cuh"
#include "tools_analysis.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"

void load_kernel_in_GPU(cuComplex* output, const float* kernel, const size_t frame_res, cudaStream_t stream)
{
   // Set the width of each element to `sizeof(float)` in bytes to copy the float data.
    // Set the pitch of the destination to `sizeof(cuComplex)` for correct alignment.
    cudaMemcpy2DAsync(output,
                      sizeof(cuComplex),      // Pitch of destination memory (width of each row in bytes)
                      kernel,
                      sizeof(float),          // Pitch of source memory (width of each row in bytes)
                      sizeof(float), // Width of data to transfer (in bytes)
                      frame_res,                      // Height of data to transfer (1 row, since it’s 1D)
                      cudaMemcpyHostToDevice,
                      stream);
}


__global__ void kernel_padding(float* output, float* input, int height, int width, int new_width, int start_x, int start_y) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int y = idx / width;
    int x = idx % width;

    if (y < height && x < width) 
    {
        output[(start_y + y) * new_width + (start_x + x)] = input[y * width + x];
    }
}

void convolution_kernel_add_padding(float* output, float* kernel, const int width, const int height, const int new_width, const int new_height, cudaStream_t stream) 
{
    //float* padded_kernel = new float[new_width * new_height];
    //std::memset(padded_kernel, 0, new_width * new_height * sizeof(float));

    int start_x = (new_width - width) / 2;
    int start_y = (new_height - height) / 2;

    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(width * height, threads);
    kernel_padding<<<blocks, threads, 0, stream>>>(output, kernel, height, width, new_width, start_x, start_y);

}