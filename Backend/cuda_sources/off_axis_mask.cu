#include "off_axis_mask.cuh"

#include "cuda_memory.cuh"
#include "tools_compute.cuh"

static __global__ void kernel_off_axis_phase_mask_and_shift_frame(const cuComplex* input,
                                                                  cuComplex* scratch,
                                                                  uint width,
                                                                  uint height,
                                                                  int x_min,
                                                                  int x_max,
                                                                  int y_min,
                                                                  int y_max,
                                                                  int shift_x,
                                                                  int shift_y)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const size_t index = static_cast<size_t>(y) * width + x;

    cuComplex value = input[index];

    const bool inside = (static_cast<int>(x) >= x_min && static_cast<int>(x) <= x_max && static_cast<int>(y) >= y_min &&
                         static_cast<int>(y) <= y_max);

    float magnitude = hypotf(value.x, value.y);
    float phase = 0.0f;

    if (inside)
    {
        phase = magnitude > 0.0f ? atan2f(value.y, value.x) : 0.0f;

        if (shift_x != 0 || shift_y != 0)
        {
            const float normalized_x =
                static_cast<float>(static_cast<int>(x) - static_cast<int>(width) / 2) / static_cast<float>(width);
            const float normalized_y =
                static_cast<float>(static_cast<int>(y) - static_cast<int>(height) / 2) / static_cast<float>(height);

            const float phase_delta =
                -2.0f * static_cast<float>(M_PI) *
                (static_cast<float>(shift_x) * normalized_x + static_cast<float>(shift_y) * normalized_y);
            phase += phase_delta;
        }
    }

    const float sine = sinf(phase);
    const float cosine = cosf(phase);

    scratch[index].x = magnitude * cosine;
    scratch[index].y = magnitude * sine;
}

void apply_off_axis_phase_mask_and_shift(const cuComplex* input,
                                         cuComplex* scratch,
                                         cuComplex* destination,
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
    dim3 lblocks((width + threads_2d - 1) / threads_2d, (height + threads_2d - 1) / threads_2d);

    for (uint batch = 0; batch < batch_size; ++batch)
    {
        const size_t offset = static_cast<size_t>(batch) * frame_res;
        const cuComplex* input_frame = input + offset;
        cuComplex* destination_frame = destination + offset;

        kernel_off_axis_phase_mask_and_shift_frame<<<lblocks, lthreads, 0, stream>>>(input_frame,
                                                                                     scratch,
                                                                                     width,
                                                                                     height,
                                                                                     x_min,
                                                                                     x_max,
                                                                                     y_min,
                                                                                     y_max,
                                                                                     shift_x,
                                                                                     shift_y);

        cudaCheckError();

        cudaXMemcpyAsync(destination_frame,
                         scratch,
                         static_cast<size_t>(frame_res) * sizeof(cuComplex),
                         cudaMemcpyDeviceToDevice,
                         stream);
    }
}
