#include "postprocessing.hh"
#include "icompute.hh"

#include "convolution.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "contrast_correction.cuh"
#include "hsv.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "map.cuh"

using holovibes::cuda_tools::CufftHandle;

namespace holovibes::compute
{
Postprocessing::Postprocessing(FunctionVector& fn_compute_vect,
                               CoreBuffersEnv& buffers,
                               const camera::FrameDescriptor& input_fd,
                               const cudaStream_t& stream)
    : gpu_kernel_buffer_()
    , cuComplex_buffer_()
    , hsv_arr_()
    , reduce_result_(1) // allocate an unique double
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , fd_(input_fd)
    , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
    , stream_(stream)
{
}

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
                                   GSH::instance().get_convo_matrix_const_ref().data(),
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

void Postprocessing::insert_convolution(bool convolution_enabled,
                                        const std::vector<float> convo_matrix,
                                        holovibes::ImgType img_type,
                                        float* gpu_postprocess_frame,
                                        float* gpu_convolution_buffer,
                                        bool divide_convolution_enabled)
{
    LOG_FUNC();

    if (!convolution_enabled || convo_matrix.empty())
        return;

    if (img_type != ImgType::Composite)
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
                                   divide_convolution_enabled,
                                   true,
                                   stream_);
            });
    }
    else
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            { convolution_composite(gpu_postprocess_frame, gpu_convolution_buffer, divide_convolution_enabled); });
    }
}

void Postprocessing::insert_renormalize(bool renorm_enabled,
                                        holovibes::ImgType img_type,
                                        float* gpu_postprocess_frame,
                                        unsigned int renorm_constant)
{
    LOG_FUNC();

    if (!renorm_enabled)
        return;

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            uint frame_res = fd_.get_frame_res();
            if (img_type == ImgType::Composite)
                frame_res *= 3;
            gpu_normalize(gpu_postprocess_frame, reduce_result_.get(), frame_res, renorm_constant, stream_);
        });
}
} // namespace holovibes::compute
