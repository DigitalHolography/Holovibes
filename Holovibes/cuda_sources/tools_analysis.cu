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

namespace
{
template <typename T>
__global__ void kernel_multiply_array_by_scalar(T* input_output, size_t size, const T scalar)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        input_output[index] *= scalar;
    }
}

template <typename T>
void multiply_array_by_scalar_caller(T* input_output, size_t size, T scalar, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_multiply_array_by_scalar<<<blocks, threads, 0, stream>>>(input_output, size, scalar);
}
}

void multiply_array_by_scalar(float* input_output, size_t size, float scalar, cudaStream_t stream)
{
    multiply_array_by_scalar_caller<float>(input_output, size, scalar, stream);
}

// CUDA kernel for computing the eigenvalues of each 2x2 Hessian matrix
__global__ void kernel_4D_eigenvalues(float *H, float *lambda_1, float *lambda_2, int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        int idx = y * cols + x; // Flattened index for 2D matrices

        // Load elements of 2x2 Hessian matrix at (y, x)
        float H11 = H[0 * rows * cols + idx]; // H(1,1)
        float H12 = H[1 * rows * cols + idx]; // H(1,2)
        float H21 = H[2 * rows * cols + idx]; // H(2,1)
        float H22 = H[3 * rows * cols + idx]; // H(2,2)

        // Compute the trace and determinant of the Hessian
        float trace = H11 + H22;
        float determinant = H11 * H22 - H12 * H21;

        // Compute the eigenvalues using the quadratic formula
        float sqrt_term = sqrtf(trace * trace - 4 * determinant);
        float eig1 = (trace - sqrt_term) / 2.0f;
        float eig2 = (trace + sqrt_term) / 2.0f;

        // Assign eigenvalues based on magnitude
        if (fabsf(eig1) <= fabsf(eig2))
        {
            lambda_1[idx] = eig1;
            lambda_2[idx] = eig2;
        } 
        else
        {
            lambda_1[idx] = eig2;
            lambda_2[idx] = eig1;
        }
    }
}

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