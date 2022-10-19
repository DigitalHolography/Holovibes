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
                               const cudaStream_t& stream,
                               AdvancedCache::Cache& advanced_cache,
                               ComputeCache::Cache& compute_cache,
                               ViewCache::Cache& view_cache)
    : gpu_kernel_buffer_()
    , cuComplex_buffer_()
    , hsv_arr_()
    , reduce_result_(1) // allocate an unique double
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , fd_(input_fd)
    , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
    , stream_(stream)
    , advanced_cache_(advanced_cache)
    , compute_cache_(compute_cache)
    , view_cache_(view_cache)
{
}

void Postprocessing::init()
{
    LOG_FUNC(compute_worker);

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
                                   GSH::instance().get_value<ConvolutionMatrix>().data(),
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
    LOG_FUNC(compute_worker);

    buffers_.gpu_convolution_buffer.reset(nullptr);
    cuComplex_buffer_.reset(nullptr);
    gpu_kernel_buffer_.reset(nullptr);
    hsv_arr_.reset(nullptr);
}

// Inserted
void Postprocessing::convolution_composite()
{
    LOG_FUNC(compute_worker);

    const size_t frame_res = fd_.get_frame_res();

    from_interweaved_components_to_distinct_components(buffers_.gpu_postprocess_frame,
                                                       hsv_arr_.get(),
                                                       frame_res,
                                                       stream_);

    convolution_kernel(hsv_arr_.get(),
                       buffers_.gpu_convolution_buffer.get(),
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_.get(),
                       compute_cache_.get_value<DivideConvolutionEnable>(),
                       true,
                       stream_);

    convolution_kernel(hsv_arr_.get() + frame_res,
                       buffers_.gpu_convolution_buffer.get(),
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_.get(),
                       compute_cache_.get_value<DivideConvolutionEnable>(),
                       true,
                       stream_);

    convolution_kernel(hsv_arr_.get() + (frame_res * 2),
                       buffers_.gpu_convolution_buffer.get(),
                       cuComplex_buffer_.get(),
                       &convolution_plan_,
                       frame_res,
                       gpu_kernel_buffer_,
                       compute_cache_.get_value<DivideConvolutionEnable>(),
                       true,
                       stream_);

    from_distinct_components_to_interweaved_components(hsv_arr_.get(),
                                                       buffers_.gpu_postprocess_frame,
                                                       frame_res,
                                                       stream_);
}

void Postprocessing::insert_convolution()
{
    LOG_FUNC(compute_worker);

    if (!compute_cache_.get_value<ConvolutionEnabled>() || compute_cache_.get_value<ConvolutionMatrix>().empty())
        return;

    if (view_cache_.get_value<ImgTypeParam>() != ImgType::Composite)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                convolution_kernel(buffers_.gpu_postprocess_frame.get(),
                                   buffers_.gpu_convolution_buffer.get(),
                                   cuComplex_buffer_.get(),
                                   &convolution_plan_,
                                   fd_.get_frame_res(),
                                   gpu_kernel_buffer_.get(),
                                   compute_cache_.get_value<DivideConvolutionEnable>(),
                                   true,
                                   stream_);
            });
    }
    else
    {
        fn_compute_vect_.conditional_push_back([=]() { convolution_composite(); });
    }
}

void Postprocessing::insert_renormalize()
{
    LOG_FUNC(compute_worker);

    if (!view_cache_.get_value<RenormEnabled>())
        return;

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            uint frame_res = fd_.get_frame_res();
            if (view_cache_.get_value<ImgTypeParam>() == ImgType::Composite)
                frame_res *= 3;
            gpu_normalize(buffers_.gpu_postprocess_frame.get(),
                          reduce_result_.get(),
                          frame_res,
                          advanced_cache_.get_value<RenormConstant>(),
                          stream_);
        });
}
} // namespace holovibes::compute
