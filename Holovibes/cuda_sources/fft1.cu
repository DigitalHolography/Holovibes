/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "fft1.cuh"
#include "transforms.cuh"
#include "unique_ptr.hh"
#include "common.cuh"
#include "cuda_memory.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;

void fft1_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks =
        map_blocks_to_problem(lens_side_size * lens_side_size, threads);

    cuComplex* square_lens;
    // In anamorphic mode, the lens is initally a square, it's then cropped to
    // be the same dimension as the frame
    if (frame_height != frame_width)
        cudaXMalloc(&square_lens,
                    lens_side_size * lens_side_size * sizeof(cuComplex));
    else
        square_lens = lens;

    kernel_quadratic_lens<<<blocks, threads, 0, stream>>>(square_lens,
                                                          lens_side_size,
                                                          lambda,
                                                          z,
                                                          pixel_size);
    cudaCheckError();

    if (frame_height != frame_width)
    {
        // Data is contiguous for a horizontal frame so a simple memcpy with an
        // offset and a limited size works
        if (frame_width > frame_height)
            cudaXMemcpyAsync(lens,
                        square_lens +
                            ((lens_side_size - frame_height) / 2) * frame_width,
                        frame_width * frame_height * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream);
        else
        {
            // For a vertical frame we need memcpy 2d to copy row by row, taking
            // the offset into account every time
            cudaSafeCall(cudaMemcpy2DAsync(
                lens,                            // Destination (frame)
                frame_width * sizeof(cuComplex), // Destination width in byte
                square_lens +
                    ((lens_side_size - frame_width) / 2), // Source (lens)
                lens_side_size * sizeof(cuComplex), // Source width in byte
                frame_width * sizeof(cuComplex),    // Destination width in byte
                                                    // (yes it's redoundant)
                frame_height, // Destination height (not in byte)
                cudaMemcpyDeviceToDevice,
                stream));
        }
        cudaXFree(square_lens);
    }
}

void fft_1(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const cuComplex* lens,
           const cufftHandle plan2D,
           const uint frame_resolution,
           const cudaStream_t stream)
{
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_resolution, threads);

    // Apply lens on multiple frames.
    kernel_apply_lens<<<blocks, threads, 0, stream>>>(input,
                                                      output,
                                                      batch_size,
                                                      frame_resolution,
                                                      lens,
                                                      frame_resolution);

    // No sync needed between kernel call and cufft call
    cudaCheckError();
    // FFT

    cufftSafeCall(cufftXtExec(plan2D, input, output, CUFFT_FORWARD));
    // Same, no sync needed since everything is executed on the stream 0

    cudaCheckError();
}
