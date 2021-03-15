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

__global__ static void filter2D_roi(cuComplex* input,
                                    const uint batch_size,
                                    const uint tl_x,
                                    const uint tl_y,
                                    const uint br_x,
                                    const uint br_y,
                                    const uint width,
                                    const uint size,
                                    const bool exclude_roi)
{
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;

    // In ROI
    if (index < size)
    {
        uint x = index % width;
        uint y = index / width;
        bool inside_roi = (y >= tl_y && y < br_y && x >= tl_x && x < br_x);
        if (inside_roi == exclude_roi)
        {
            for (uint i = 0; i < batch_size; ++i)
            {
                const uint batch_index = index + i * size;

                input[batch_index] = make_cuComplex(0, 0);
            }
        }
    }
}

__global__ void kernel_filter2D_BandPass(cuComplex* input,
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

void filter2D_BandPass(cuComplex* input,
                       cuComplex* tmp_buffer,
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

    if (!zone.area() || !subzone.area())
        return;

    shift_corners(input, batch_size, desc.width, desc.height, stream);

    kernel_filter2D_BandPass<<<blocks, threads, 0, stream>>>(
        input,
        batch_size,
        zone.topLeft().x(),
        zone.topLeft().y(),
        zone.bottomRight().x(),
        zone.bottomRight().y(),
        subzone.topLeft().x(),
        subzone.topLeft().y(),
        subzone.bottomRight().x(),
        subzone.bottomRight().y(),
        desc.width,
        size);
    cudaCheckError();

    shift_corners(input, tmp_buffer, batch_size, desc.width, desc.height, stream);

    circ_shift<<<blocks, threads, 0, stream>>>(tmp_buffer,
                                               input,
                                               batch_size,
                                               zone.center().x(),
                                               zone.center().y(),
                                               desc.width,
                                               desc.height,
                                               size);
    cudaCheckError();

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}

void filter2D(cuComplex* input,
              cuComplex* tmp_buffer,
              const uint batch_size,
              const cufftHandle plan2d,
              const holovibes::units::RectFd& r,
              const FrameDescriptor& desc,
              const bool exclude_roi,
              const cudaStream_t stream)
{
    uint threads = THREADS_128;
    uint blocks = map_blocks_to_problem(desc.frame_res(), threads);
    uint size = desc.width * desc.height;

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    if (!r.area())
        return;

    shift_corners(input, batch_size, desc.width, desc.height, stream);

    filter2D_roi<<<blocks, threads, 0, stream>>>(input,
                                                 batch_size,
                                                 r.topLeft().x(),
                                                 r.topLeft().y(),
                                                 r.bottomRight().x(),
                                                 r.bottomRight().y(),
                                                 desc.width,
                                                 size,
                                                 exclude_roi);
    cudaCheckError();

    shift_corners(input, tmp_buffer, batch_size, desc.width, desc.height, stream);

    circ_shift<<<blocks, threads, 0, stream>>>(tmp_buffer,
                                               input,
                                               batch_size,
                                               r.center().x(),
                                               r.center().y(),
                                               desc.width,
                                               desc.height,
                                               size);
    cudaCheckError();

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));
}
