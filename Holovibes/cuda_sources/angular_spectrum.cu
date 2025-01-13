#include "angular_spectrum.cuh"

#include <cufftXt.h>

#include "apply_mask.cuh"
#include "cuda_memory.cuh"
#include "frame_desc.hh"
#include "masks.cuh"
#include "shift_corners.cuh"
#include "tools_compute.cuh"

using camera::FrameDescriptor;

// void angular_spectrum_lens(cuComplex* lens,
//                            const uint lens_side_size,
//                            const uint frame_height,
//                            const uint frame_width,
//                            const float lambda,
//                            const float z,
//                            const float pixel_size,
//                            const cudaStream_t stream)
// {
//     const uint threads_2d = get_max_threads_2d();
//     const dim3 lthreads(threads_2d, threads_2d);
//     const dim3 lblocks(lens_side_size / threads_2d, lens_side_size / threads_2d);

//     cuComplex* square_lens;
//     // In anamorphic mode, the lens is initally a square, it's then cropped to
//     // be the same dimension as the frame
//     if (frame_height != frame_width)
//         cudaXMalloc(&square_lens, lens_side_size * lens_side_size * sizeof(cuComplex));
//     else
//         square_lens = lens;

//     kernel_spectral_lens<<<lblocks, lthreads, 0, stream>>>(square_lens, lens_side_size, lambda, z, pixel_size);
//     cudaCheckError();

//     // if (frame_height != frame_width)
//     // {
//     //     cudaXMemcpyAsync(lens,
//     //                      square_lens + ((lens_side_size - frame_height) / 2) * frame_width,
//     //                      frame_width * frame_height * sizeof(cuComplex),
//     //                      cudaMemcpyDeviceToDevice,
//     //                      stream);
//     //     cudaXFree(square_lens);
//     // }
//     if (frame_height != frame_width)
//     {
//         // Data is contiguous for a horizontal frame so a simple memcpy with an
//         // offset and a limited size works
//         if (frame_width > frame_height)
//             cudaXMemcpyAsync(lens,
//                              square_lens + ((lens_side_size - frame_height) / 2) * frame_width,
//                              frame_width * frame_height * sizeof(cuComplex),
//                              cudaMemcpyDeviceToDevice,
//                              stream);
//         else
//         {
//             // For a vertical frame we need memcpy 2d to copy row by row, taking
//             // the offset into account every time
//             cudaSafeCall(cudaMemcpy2DAsync(lens,                            // Destination (frame)
//                                            frame_width * sizeof(cuComplex), // Destination width in byte
//                                            square_lens + ((lens_side_size - frame_width) / 2), // Source (lens)
//                                            lens_side_size * sizeof(cuComplex),                 // Source width in
//                                            byte frame_width * sizeof(cuComplex), // Destination width in byte
//                                                                             // (yes it's redoundant)
//                                            frame_height,                    // Destination height (not in byte)
//                                            cudaMemcpyDeviceToDevice,
//                                            stream));
//         }
//         cudaXFree(square_lens);
//     }
// }

__global__ void computeKernel(cuFloatComplex* kernel, int Nx, int Ny, float z, float lambda, float x_step, float y_step)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < Nx && y < Ny)
    {
        float u_step = 1.0f / (Nx * x_step);
        float v_step = 1.0f / (Ny * y_step);

        float u = (x - (Nx / 2)) * u_step;
        float v = (y - (Ny / 2)) * v_step;

        float tmp = 1.0f - (lambda * lambda * (u * u + v * v));
        if (tmp < 0.0f)
            tmp = 0.0f; // Ensure positivity under sqrt
        float phase = 2.0f * M_PI * z / lambda * sqrtf(tmp);

        // Store result as complex exponential
        kernel[y * Nx + x] = make_cuFloatComplex(cosf(phase), sinf(phase));
    }
}

// Host function to launch the CUDA kernel
void angular_spectrum_lens(cuFloatComplex* d_kernel,
                           int Nx,
                           int Ny,
                           float z,
                           float lambda,
                           float x_step,
                           float y_step,
                           const cudaStream_t stream)
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x, (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    computeKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_kernel, Nx, Ny, z, lambda, x_step, y_step);
    cudaXStreamSynchronize(stream);
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
