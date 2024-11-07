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

// CUDA kernel to prepare H hessian matrices
__global__ void kernel_prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        // Prepare the 2x2 submatrix for point `index`
        output[index * 3 + 0] = ixx[index];
        output[index * 3 + 1] = ixy[index];
        output[index * 3 + 2] = iyy[index];
    }
}

void prepare_hessian(float* output, const float* ixx, const float* ixy, const float* iyy, const int size, cudaStream_t stream)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    kernel_prepare_hessian<<<numBlocks, blockSize, 0, stream>>>(output, ixx, ixy, iyy, size);
}

__global__ void kernel_compute_eigen(float* H, int size, float* lambda1, float* lambda2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        double a = H[index * 3], b = H[index * 3 + 1], d = H[index * 3 + 2];
        double trace = a + d;
        double determinant = a * d - b * b;
        double discriminant = trace * trace - 4 * determinant;
        if (discriminant >= 0)
        {
            double eig1 = (trace + std::sqrt(discriminant)) / 2;
            double eig2 = (trace - std::sqrt(discriminant)) / 2;
            if (std::abs(eig1) < std::abs(eig2))
            {
                lambda1[index] = eig1;
                lambda2[index] = eig2;
            }
            else
            {
                lambda1[index] = eig2;
                lambda2[index] = eig1;
            }
        }
    }
}

void compute_eigen_values(float* H, int size, float* lambda1, float* lambda2, cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    kernel_compute_eigen<<<blocks, threads, 0, stream>>>(H, size, lambda1, lambda2);
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

void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
            {
                outFile << " "; // Separate values in a row by a space
            }
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

void print_in_file(float* input, uint size, std::string filename, cudaStream_t stream)
{
    if (input == nullptr)
    {
        return;
    }
    float* result = new float[size];
    cudaXMemcpyAsync(result,
                        input,
                        size * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream);
    write1DFloatArrayToFile(result,
                            sqrt(size),
                            sqrt(size),
                            "test_" + filename + ".txt");
}

__global__ void
kernel_apply_diaphragm_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared > radius_squared)
            output[index] = 0;
    }
}

void apply_diaphragm_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_diaphragm_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_compute_circle_mask(float* output, short width, short height, float center_X, float center_Y, float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        float distance_squared = (x - center_X) * (x - center_X) + (y - center_Y) * (y - center_Y);
        float radius_squared = radius * radius;

        // If the point is inside the circle set the value to 1.
        if (distance_squared <= radius_squared)
            output[index] = 1;
        else
            output[index] = 0;
    }
}

void compute_circle_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_compute_circle_mask<<<lblocks, lthreads, 0, stream>>>(output, width, height, center_X, center_Y, radius);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_apply_mask_and(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        output[y * width + x] *= input[y * width + x];
    }
}

void apply_mask_and(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_and<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

__global__ void
kernel_apply_mask_or(float* output, const float* input, short width, short height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    if (x < width && y < height)
    {
        output[y * width + x] = (input[y * width + x] != 0.f) ? 1.f : output[y * width + x];
    }
}

void apply_mask_or(float* output,
                       const float* input,
                       const short width,
                       const short height,
                       const cudaStream_t stream)
{
    // Setting up the parallelisation.
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);

    kernel_apply_mask_or<<<lblocks, lthreads, 0, stream>>>(output, input, width, height);

    cudaXStreamSynchronize(stream);
    cudaCheckError();
}