#include "fft2.cuh"
#include "transforms.cuh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "apply_mask.cuh"
#include "shift_corners.cuh"

#include <cufftXt.h>

using camera::FrameDescriptor;

void fft2_lens(cuComplex* lens,
               const uint lens_side_size,
               const uint frame_height,
               const uint frame_width,
               const float lambda,
               const float z,
               const float pixel_size,
               const cudaStream_t stream)
{
    const uint threads_2d = get_max_threads_2d();
    const dim3 lthreads(threads_2d, threads_2d);
    const dim3 lblocks(lens_side_size / threads_2d,
                       lens_side_size / threads_2d);

    cuComplex* square_lens;
    // In anamorphic mode, the lens is initally a square, it's then cropped to
    // be the same dimension as the frame
    if (frame_height != frame_width)
        cudaXMalloc(&square_lens,
                    lens_side_size * lens_side_size * sizeof(cuComplex));
    else
        square_lens = lens;

    kernel_spectral_lens<<<lblocks, lthreads, 0, stream>>>(square_lens,
                                                           lens_side_size,
                                                           lambda,
                                                           z,
                                                           pixel_size);
    cudaCheckError();

    if (frame_height != frame_width)
    {
        cudaXMemcpyAsync(lens,
                         square_lens + ((lens_side_size - frame_height) / 2) *
                                           frame_width,
                         frame_width * frame_height * sizeof(cuComplex),
                         cudaMemcpyDeviceToDevice,
                         stream);
        cudaXFree(square_lens);
    }
}

void fft_2(cuComplex* input,
           cuComplex* output,
           const uint batch_size,
           const cuComplex* lens,
           const cufftHandle plan2d,
           const FrameDescriptor& fd,
           const cudaStream_t stream)
{
    const uint frame_resolution = fd.frame_res();
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_resolution, threads);

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    // Lens and Mask already shifted
    // thus we don't have to shift the 'input' buffer each time
    apply_mask(input, lens, output, frame_resolution, batch_size, stream);

    cudaCheckError();

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));

    kernel_complex_divide<<<blocks, threads, 0, stream>>>(
        input,
        frame_resolution,
        static_cast<float>(frame_resolution),
        batch_size);
    cudaCheckError();
}