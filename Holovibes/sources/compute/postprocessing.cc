#include "postprocessing.hh"
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

void Postprocessing::init()
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

    hsv_arr_.resize(frame_res * 3);
}

void Postprocessing::dispose()
{
    LOG_FUNC();

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gpu_kernel_buffer_.reset(nullptr);
    hsv_arr_.reset(nullptr);
}

// Inserted
void Postprocessing::convolution_composite(float* gpu_postprocess_frame,
                                           float* gpu_convolution_buffer,
                                           bool divide_convolution_enabled)
{
    LOG_FUNC();

    const size_t frame_res = fd_.get_frame_res();

    from_interweaved_components_to_distinct_components(gpu_postprocess_frame, hsv_arr_.get(), frame_res, stream_);

    convolution_kernel(hsv_arr_.get(),
                       gpu_convolution_buffer,
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_.get(),
                       divide_convolution_enabled,
                       true,
                       stream_);

    convolution_kernel(hsv_arr_.get() + frame_res,
                       gpu_convolution_buffer,
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_.get(),
                       divide_convolution_enabled,
                       true,
                       stream_);

    convolution_kernel(hsv_arr_.get() + (frame_res * 2),
                       gpu_convolution_buffer,
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_,
                       divide_convolution_enabled,
                       true,
                       stream_);

    from_distinct_components_to_interweaved_components(hsv_arr_.get(), gpu_postprocess_frame, frame_res, stream_);
}

void Postprocessing::insert_convolution(float* gpu_postprocess_frame,
                                        float* gpu_convolution_buffer)
{
    LOG_FUNC();

    if (!setting<settings::ConvolutionEnabled>() || setting<settings::ConvolutionMatrix>().empty())
        return;

    if (setting<settings::ImageType>() != ImgType::Composite)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                convolution_kernel(gpu_postprocess_frame,
                                   gpu_convolution_buffer,
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gpu_kernel_buffer_.get(),
                                   setting<settings::DivideConvolutionEnabled>(),
                                   true,
                                   stream_);
            });
    }
    else
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                convolution_composite(gpu_postprocess_frame,
                                      gpu_convolution_buffer,
                                      setting<settings::DivideConvolutionEnabled>());
            });
    }
}

void Postprocessing::insert_renormalize(float* gpu_postprocess_frame)
{
    LOG_FUNC();

    if (!setting<settings::RenormEnabled>())
        return;

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            uint frame_res = fd_.get_frame_res();
            if (setting<settings::ImageType>() == ImgType::Composite)
                frame_res *= 3;
            gpu_normalize(gpu_postprocess_frame,
                          reduce_result_.get(),
                          frame_res,
                          setting<settings::RenormConstant>(),
                          stream_);
        });
}
} // namespace holovibes::compute
