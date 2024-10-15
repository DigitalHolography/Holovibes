#include "shift_corners.cuh"

#include "hardware_limits.hh"
#include "common.cuh"

namespace
{
template <typename T>
__global__ void kernel_shift_corners(T* buffer, const uint batch_size, const uint size_x, const uint size_y)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    const uint size_x2 = size_x / 2;
    const uint size_y2 = size_y / 2;

    // Superior-left quarter of the matrix
    if (x >= size_x2 || y >= size_y2)
        return;

    // double swap:
    // top-left <=> bottom-right
    // top-right <=> bottom-left

    const uint top_left = y * size_x + x;
    const uint top_right = y * size_x + (x + size_x2);
    const uint bot_right = (y + size_x2) * size_x + (x + size_x2);
    const uint bot_left = (y + size_y2) * size_x + x;

    for (uint i = 0; i < batch_size; ++i)
    {
        const uint index = i * size_x * size_y;

        T tmp = buffer[index + top_left];
        buffer[index + top_left] = buffer[index + bot_right];
        buffer[index + bot_right] = tmp;

        tmp = buffer[index + top_right];
        buffer[index + top_right] = buffer[index + bot_left];
        buffer[index + bot_left] = tmp;
    }
}

template <typename T>
void shift_corners_caller(
    T* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(static_cast<ushort>(std::ceil((size_x / 2) / static_cast<float>(lthreads.x))),
                 static_cast<ushort>(std::ceil((size_y / 2) / static_cast<float>(lthreads.y))));

    kernel_shift_corners<T><<<lblocks, lthreads, 0, stream>>>(input, batch_size, size_x, size_y);
    cudaCheckError();
}
} // namespace

void shift_corners(
    float3* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<float3>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(float* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<float>(input, batch_size, size_x, size_y, stream);
}

void shift_corners(
    cuComplex* input, const uint batch_size, const uint size_x, const uint size_y, const cudaStream_t stream)
{
    shift_corners_caller<cuComplex>(input, batch_size, size_x, size_y, stream);
}
