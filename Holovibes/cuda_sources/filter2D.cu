/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "filter2D.cuh"
#include "shift_corners.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;

__global__ void kernel_filter2D(cuComplex* input,
                                const uint batch_size,
                                const uint zone_tl_x,
                                const uint zone_tl_y,
                                const uint zone_br_x,
                                const uint zone_br_y,
                                const uint subzone_tl_x,
                                const uint subzone_tl_y,
                                const uint subzone_br_x,
                                const uint subzone_br_y,
                                const uint width,
                                const uint size)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    // In ROI
    if (index < size)
    {
        uint x = index % width;
        uint y = index / width;
        bool inside_zone = (y >= zone_tl_y && y < zone_br_y && x >= zone_tl_x &&
                            x < zone_br_x);
        bool inside_sub_zone = (y >= subzone_tl_y && y < subzone_br_y &&
                                x >= subzone_tl_x && x < subzone_br_x);
        bool outside_selection = !inside_zone || inside_sub_zone;
        if (outside_selection)
        {
            for (uint i = 0; i < batch_size; ++i)
            {
                const uint batch_index = index + i * size;

                input[batch_index] = make_cuComplex(0, 0);
            }
        }
    }
}

__global__ void kernel_gen_filter2d_squares_mask(float *in_out,
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

        const bool inside_zone = (y >= zone_tl_y && y < zone_br_y && x >= zone_tl_x &&
                                  x < zone_br_x);
        const bool inside_sub_zone = (y >= subzone_tl_y && y < subzone_br_y &&
                                      x >= subzone_tl_x && x < subzone_br_x);
        const bool outside_selection = !inside_zone || inside_sub_zone;

        in_out[index] = !outside_selection;
    }
}

void filter2D(cuComplex* input,
              const uint batch_size,
              const cufftHandle plan2d,
              const holovibes::units::RectFd& zone,
              const holovibes::units::RectFd& subzone,
              const FrameDescriptor& desc,
              const cudaStream_t stream)
{
    uint threads = THREADS_128;
    uint blocks = map_blocks_to_problem(desc.frame_res(), threads);
    uint size = desc.width * desc.height;

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    shift_corners(input, batch_size, desc.width, desc.height, stream);

    // fft + mask + fft = filter2d => misuse of language
    kernel_filter2D<<<blocks, threads, 0, stream>>>(input,
                                                    batch_size,
                                                    zone.topLeft().x().get(),
                                                    zone.topLeft().y().get(),
                                                    zone.bottomRight().x().get(),
                                                    zone.bottomRight().y().get(),
                                                    subzone.topLeft().x().get(),
                                                    subzone.topLeft().y().get(),
                                                    subzone.bottomRight().x().get(),
                                                    subzone.bottomRight().y().get(),
                                                    desc.width,
                                                    size);
    cudaCheckError();

    shift_corners(input, batch_size, desc.width, desc.height, stream);

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}

void gen_filter2d_squares_mask(float *in_out,
                              const uint width,
                              const uint height,
                              const uint sq_in_radius,
                              const uint sq_out_radius,
                              const cudaStream_t stream)
{
    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(width / threads_2d, height / threads_2d);

    kernel_gen_filter2d_squares_mask<<<lblocks, lthreads, 0, stream>>>(in_out,
                                                                    width * height,
                                                                    width / 2,
                                                                    height / 2,
                                                                    sq_in_radius,
                                                                    sq_out_radius);
}