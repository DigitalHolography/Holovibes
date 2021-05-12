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

    // Mask already shifted in update_filter2d_squares_mask()
    // thus we don't have to shift the 'input' buffer each time
    apply_mask(input, mask, input, size, batch_size, stream);

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}

__global__ void kernel_update_filter2d_squares_mask(float* in_out,
                                                    const uint size,
                                                    const uint middle_x,
                                                    const uint middle_y,
                                                    const uint sq_in_radius,
                                                    const uint sq_out_radius)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint index = y * blockDim.x * gridDim.x + x;

    if (index < size)
    {
        const uint zone_tl_x = middle_x - sq_out_radius;
        const uint zone_tl_y = middle_y - sq_out_radius;
        const uint zone_br_x = middle_x + sq_out_radius;
        const uint zone_br_y = middle_y + sq_out_radius;
        const uint subzone_tl_x = middle_x - sq_in_radius;
        const uint subzone_tl_y = middle_y - sq_in_radius;
        const uint subzone_br_x = middle_x + sq_in_radius;
        const uint subzone_br_y = middle_y + sq_in_radius;

        const bool inside_zone = (y >= zone_tl_y && y < zone_br_y &&
                                  x >= zone_tl_x && x < zone_br_x);
        const bool inside_sub_zone = (y >= subzone_tl_y && y < subzone_br_y &&
                                      x >= subzone_tl_x && x < subzone_br_x);
        const bool outside_selection = !inside_zone || inside_sub_zone;

        in_out[index] = !outside_selection;
    }
}

void update_filter2d_squares_mask(float* in_out,
                                  const uint width,
                                  const uint height,
                                  const uint sq_in_radius,
                                  const uint sq_out_radius,
                                  const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(width / threads_2d, height / threads_2d);

    kernel_update_filter2d_squares_mask<<<lblocks, lthreads, 0, stream>>>(
        in_out,
        width * height,
        width / 2,
        height / 2,
        sq_in_radius,
        sq_out_radius);

    shift_corners(in_out, 1, width, height, stream);
}