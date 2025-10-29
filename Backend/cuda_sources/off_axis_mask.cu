#include "off_axis_mask.cuh"

#include "cuda_memory.cuh"
#include "tools_compute.cuh"

static __global__ void kernel_off_axis_phase_mask_and_shift(cuComplex* output,
                                                            const cuComplex* input,
                                                            uint width,
                                                            uint height,
                                                            uint frame_res,
                                                            int x_min,
                                                            int x_max,
                                                            int y_min,
                                                            int y_max,
                                                            int shift_x,
                                                            int shift_y)
{
    const uint batch = blockIdx.z;
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const size_t base_index = static_cast<size_t>(batch) * frame_res;
    const size_t index = base_index + static_cast<size_t>(y) * width + x;

    cuComplex value = input[index];

    const bool inside = (static_cast<int>(x) >= x_min && static_cast<int>(x) <= x_max && static_cast<int>(y) >= y_min &&
                         static_cast<int>(y) <= y_max);

    if (!inside)
    {
        const float magnitude = sqrtf(value.x * value.x + value.y * value.y);
        value.x = magnitude;
        value.y = 0.0f;
    }

    int new_x = static_cast<int>(x) + shift_x;
    int new_y = static_cast<int>(y) + shift_y;

    new_x %= static_cast<int>(width);
    new_y %= static_cast<int>(height);

    if (new_x < 0)
        new_x += static_cast<int>(width);
    if (new_y < 0)
        new_y += static_cast<int>(height);

    const size_t dest_index = base_index + static_cast<size_t>(new_y) * width + static_cast<uint>(new_x);
    output[dest_index] = value;
}

void apply_off_axis_phase_mask_and_shift(const cuComplex* input,
                                         cuComplex* output,
                                         uint width,
                                         uint height,
                                         uint frame_res,
                                         uint batch_size,
                                         int x_min,
                                         int x_max,
                                         int y_min,
                                         int y_max,
                                         int shift_x,
                                         int shift_y,
                                         const cudaStream_t stream)
{
    if (batch_size == 0 || width == 0 || height == 0)
        return;

    const uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks((width + threads_2d - 1) / threads_2d, (height + threads_2d - 1) / threads_2d, batch_size);

    kernel_off_axis_phase_mask_and_shift<<<lblocks, lthreads, 0, stream>>>(output,
                                                                           input,
                                                                           width,
                                                                           height,
                                                                           frame_res,
                                                                           x_min,
                                                                           x_max,
                                                                           y_min,
                                                                           y_max,
                                                                           shift_x,
                                                                           shift_y);

    cudaCheckError();
}
