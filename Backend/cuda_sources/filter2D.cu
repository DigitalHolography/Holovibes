#include "filter2D.cuh"
#include "shift_corners.cuh"
#include "apply_mask.cuh"
#include "tools_conversion.cuh"
#include "cuda_memory.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;

/*
void filter2D(cuComplex* input,
              const float* mask,
              const uint batch_size,
              const cufftHandle plan2d,
              const uint size,
              const cudaStream_t stream)
{
    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    // Mask already shifted in update_filter2d_circles_mask()
    // thus we don't have to shift the 'input' buffer each time
    apply_mask(input, mask, size, batch_size, stream);

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}
*/

void filter2D(cuComplex* input,
              const float* mask,
              cuComplex* output,
              bool store_frame,
              const uint batch_size,
              const cufftHandle plan2d,
              const uint width,
              const uint length,
              const cudaStream_t stream)
{
    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    // Mask already shifted in update_filter2d_circles_mask()
    // thus we don't have to shift the 'input' buffer each time
    apply_mask(input, mask, width * length, batch_size, stream);
    if (store_frame)
    {
        cudaXMemcpyAsync(output, input, width * length * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream);
    }

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}

static __global__ void kernel_update_filter2d_circles_mask(float* in_out,
                                                           const uint size,
                                                           const uint width,
                                                           const uint height,
                                                           const uint radius_low,
                                                           const uint radius_high,
                                                           const uint smooth_low,
                                                           const uint smooth_high)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = y * blockDim.x * gridDim.x + x;

    if (index < size)
    {
        float a = 0.0f, b = 0.0f;

        // Relatives values to the center of the image
        const float r_x = (float)x - width / 2;
        const float r_y = (float)y - height / 2;

        const float length = hypotf(r_x, r_y);

        // A: big disc
        if (length < radius_high)
            a = 1.0f;
        else if (length < radius_high + smooth_high)
            a = cosf(((length - radius_high) / (float)(smooth_high)) * M_PI_2);

        // B: small disc
        if (length < radius_low)
            b = 1.0f;
        else if (length < radius_low + smooth_low)
            b = cosf(((length - radius_low) / (float)(smooth_low)) * M_PI_2);

        // pixel = A * (1 - B)
        in_out[index] = a * (1 - b);
    }
}

void update_filter2d_circles_mask(float* in_out,
                                  const uint width,
                                  const uint height,
                                  const uint radius_low,
                                  const uint radius_high,
                                  const uint smooth_low,
                                  const uint smooth_high,
                                  const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(width / threads_2d, height / threads_2d);

    kernel_update_filter2d_circles_mask<<<lblocks, lthreads, 0, stream>>>(in_out,
                                                                          width * height,
                                                                          width,
                                                                          height,
                                                                          radius_low,
                                                                          radius_high,
                                                                          smooth_low,
                                                                          smooth_high);

    shift_corners(in_out, 1, width, height, stream);
}
