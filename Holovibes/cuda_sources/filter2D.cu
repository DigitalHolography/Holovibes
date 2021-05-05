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
