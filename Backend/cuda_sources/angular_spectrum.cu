#include "angular_spectrum.cuh"

#include <cufftXt.h>

#include "apply_mask.cuh"
#include "cuda_memory.cuh"
#include "frame_desc.hh"
#include "masks.cuh"
#include "shift_corners.cuh"
#include "tools_compute.cuh"

using camera::FrameDescriptor;

void angular_spectrum_lens(cuFloatComplex* output,
                           const int Nx,
                           const int Ny,
                           const float z,
                           const float lambda,
                           const float x_step,
                           const float y_step,
                           const cudaStream_t stream)
{

    uint threads_2d = get_max_threads_2d();
    dim3 lthreads(threads_2d, threads_2d);
    dim3 lblocks(1 + (Nx - 1) / threads_2d, 1 + (Ny - 1) / threads_2d);

    kernel_spectral_lens<<<lblocks, lthreads, 0, stream>>>(output, Nx, Ny, z, lambda, x_step, y_step);
    cudaXStreamSynchronize(stream);
    cudaCheckError();
}

void angular_spectrum(cuComplex* input,
                      cuComplex* output,
                      const uint batch_size,
                      const cuComplex* lens,
                      cuComplex* mask_output,
                      bool store_frame,
                      const cufftHandle plan2d,
                      const FrameDescriptor& fd,
                      const cudaStream_t stream)
{
    const uint frame_res = fd.get_frame_res();
    const uint threads = get_max_threads_1d();
    const uint blocks = map_blocks_to_problem(frame_res, threads);

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_FORWARD));

    // Lens and Mask already shifted
    // thus we don't have to shift the 'input' buffer each time
    apply_mask(input, lens, output, frame_res, batch_size, stream);
    if (store_frame)
    {
        cudaXMemcpyAsync(mask_output, input, frame_res * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream);
    }

    cudaCheckError();

    cufftSafeCall(cufftXtExec(plan2d, input, input, CUFFT_INVERSE));

    kernel_complex_divide<<<blocks, threads, 0, stream>>>(input, frame_res, static_cast<float>(frame_res), batch_size);
    cudaCheckError();
}
