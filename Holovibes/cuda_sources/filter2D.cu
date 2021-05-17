/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "filter2D.cuh"
#include "shift_corners.cuh"
#include "apply_mask.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;

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

static __device__ float pow2(float x) { return (x * x); }

static __device__ float length(const float x, const float y)
{
    return (sqrtf(pow2(x) + pow2(y)));
}

static __global__ void
kernel_update_filter2d_circles_mask(float* in_out,
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
        float a, b;

        // Relatives values to the center of the image
        const float r_x = (float)x - width / 2;
        const float r_y = (float)y - height / 2;

        // A: big disc
        if (pow2(length(r_x, r_y)) < pow2(radius_high))
            a = 1.0f;
        else if (length(r_x, r_y) < radius_high + smooth_high)
            a = cosf(((length(r_x, r_y) - radius_high) / (float)(smooth_high)) *
                     M_PI_2);
        else
            a = 0.0f;

        // B: small disc
        if (pow2(length(r_x, r_y)) < pow2(radius_low))
            b = 1.0f;
        else if (length(r_x, r_y) < radius_low + smooth_low)
            b = cosf(((length(r_x, r_y) - radius_low) / (float)(smooth_low)) *
                     M_PI_2);
        else
            b = 0.0f;

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

    kernel_update_filter2d_circles_mask<<<lblocks, lthreads, 0, stream>>>(
        in_out,
        width * height,
        width,
        height,
        radius_low,
        radius_high,
        smooth_low,
        smooth_high);

    shift_corners(in_out, 1, width, height, stream);
}