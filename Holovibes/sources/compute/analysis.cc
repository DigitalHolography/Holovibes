#include "analysis.hh"
#include "icompute.hh"

#include "convolution.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "contrast_correction.cuh"
#include "tools_hsv.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"

using holovibes::cuda_tools::CufftHandle;

namespace holovibes::compute
{

void Analysis::init()
{
    LOG_FUNC();
    const size_t frame_res = fd_.get_frame_res();

    // No need for memset here since it will be completely overwritten by
    // cuComplex values
    buffers_.gpu_convolution_buffer.resize(frame_res);

    // No need for memset here since it will be memset in the actual convolution
    cuComplex_buffer_.resize(frame_res);

    gpu_kernel_buffer_.resize(frame_res);
    cudaXMemsetAsync(gpu_kernel_buffer_.get(), 0, frame_res * sizeof(cuComplex), stream_);
    cudaSafeCall(cudaMemcpy2DAsync(gpu_kernel_buffer_.get(),
                                   sizeof(cuComplex),
                                   setting<settings::ConvolutionMatrix>().data(),
                                   sizeof(float),
                                   sizeof(float),
                                   frame_res,
                                   cudaMemcpyHostToDevice,
                                   stream_));

    constexpr uint batch_size = 1; // since only one frame.
    // We compute the FFT of the kernel, once, here, instead of every time the
    // convolution subprocess is called
    shift_corners(gpu_kernel_buffer_.get(), batch_size, fd_.width, fd_.height, stream_);
    cufftSafeCall(cufftExecC2C(convolution_plan_, gpu_kernel_buffer_.get(), gpu_kernel_buffer_.get(), CUFFT_FORWARD));
}

void Analysis::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gpu_kernel_buffer_.reset(nullptr);
}

void Analysis::insert_show_artery()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            if (setting<settings::ImageType>() == ImgType::Moments_0 && setting<settings::ArteryMaskEnabled>() == true)
            {
                // apply_flat_field_correction(buffers_.gpu_postprocess_frame,
                //                             fd_.width,
                //                             0.07 * fd_.width,
                //                             0.25f,
                //                             stream_);
                // TODO do convolution with 256 gauss
            }
        });
}
} // namespace holovibes::compute