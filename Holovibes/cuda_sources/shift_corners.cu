#include "shift_corners.cuh"

#include "hardware_limits.hh"
#include "common.cuh"

namespace
{
template <typename T>
__global__ void
kernel_shift_corners(T* output, const T* input, const uint batch_size, const uint size_x, const uint size_y)
{
    const uint i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint j = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = j * (size_x) + i;
    uint ni = 0;
    uint nj = 0;
    uint nindex = 0;

    const uint size_x2 = size_x / 2;
    const uint size_y2 = size_y / 2;

    // Superior half of the matrix
    if (j < size_y2)
    {
        // Left superior quarter of the matrix
        if (i < size_x2)
            ni = i + size_x2;
        else // Right superior quarter
            ni = i - size_x2;
        nj = j + size_y2;
        nindex = nj * size_x + ni;

        for (uint i = 0; i < batch_size; ++i)
        {
            const uint batch_index = index + i * size_x * size_y;
            const uint batch_nindex = nindex + i * size_x * size_y;

            // Allows output = input
            T tmp = input[batch_nindex];
            output[batch_nindex] = input[batch_index];
            output[batch_index] = tmp;
        }
    }
}

template <typename T>
void shift_corners_caller(
    T* output, const T* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (size_x - 1) / threads_2d, 1 + (size_y - 1) / threads_2d);

    kernel_shift_corners<T><<<lblocks, lthreads, 0, stream>>>(output, input, batch_size, size_x, size_y);
    cudaCheckError();
}

template <typename T>
void shift_corners_caller(
    T* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(static_cast<ushort>(std::ceil(size_x / static_cast<float>(lthreads.x))),
                 static_cast<ushort>(std::ceil(size_y / static_cast<float>(lthreads.y))));

    kernel_shift_corners<T><<<lblocks, lthreads, 0, stream>>>(input, input, batch_size, size_x, size_y);
    cudaCheckError();
}
} // namespace

void shift_corners(
    float3* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<float3>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(float3* output,
                   const float3* input,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream)
{
    shift_corners_caller<float3>(output, input, batch_size, size_x, size_y, stream);
}

void shift_corners(float* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<float>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(float* output,
                   const float* input,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream)
{
    shift_corners_caller<float>(output, input, batch_size, size_x, size_y, stream);
}

void shift_corners(
    cuComplex* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<cuComplex>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(cuComplex* output,
                   const cuComplex* input,
                   const uint batch_size,
                   const uint size_x,
                   const uint size_y,
                   const cudaStream_t stream)
{
    shift_corners_caller<cuComplex>(output, input, batch_size, size_x, size_y, stream);
}
