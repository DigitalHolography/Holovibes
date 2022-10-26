#include "image_accumulation.hh"

#include "rect.hh"
#include "power_of_two.hh"
#include "cufft_handle.hh"

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "icompute.hh"
#include "logger.hh"
#include "cuda_memory.cuh"

namespace holovibes
{
using cuda_tools::CufftHandle;
using cuda_tools::UniquePtr;
} // namespace holovibes

namespace holovibes::compute
{
ImageAccumulation::ImageAccumulation(FunctionVector& fn_compute_vect,
                                     ImageAccEnv& image_acc_env,
                                     const CoreBuffersEnv& buffers,
                                     const camera::FrameDescriptor& fd,
                                     const cudaStream_t& stream,
                                     ViewCache::Cache& view_cache)
    : fn_compute_vect_(fn_compute_vect)
    , image_acc_env_(image_acc_env)
    , buffers_(buffers)
    , fd_(fd)
    , stream_(stream)
    , view_cache_(view_cache)
//, cache_(cache)
{
}

void ImageAccumulation::insert_image_accumulation()
{
    LOG_FUNC();

    insert_compute_average();
    insert_copy_accumulation_result();
}

void ImageAccumulation::compute_average(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                        float* gpu_input_frame,
                                        float* gpu_ouput_average_frame,
                                        const unsigned int image_acc_level,
                                        const size_t frame_res)
{
    // LOG-USELESS LOG_FUNC(image_acc_level, frame_res);

    if (gpu_accumulation_queue)
    {
        // Enqueue the computed frame in the accumulation queue
        gpu_accumulation_queue->enqueue(gpu_input_frame, stream_);

        // Compute the average and store it in the output frame
        accumulate_images(static_cast<float*>(gpu_accumulation_queue->get_data()),
                          gpu_ouput_average_frame,
                          gpu_accumulation_queue->get_size(),
                          gpu_accumulation_queue->get_max_size(),
                          image_acc_level,
                          frame_res,
                          stream_);
    }
}

void ImageAccumulation::insert_compute_average()
{
    LOG_FUNC();

    auto compute_average_lambda = [&]()
    {
        // XY view
        if (image_acc_env_.gpu_accumulation_xy_queue && view_cache_.get_value<ViewXY>().is_image_accumulation_enabled())
            compute_average(image_acc_env_.gpu_accumulation_xy_queue,
                            buffers_.gpu_postprocess_frame.get(),
                            image_acc_env_.gpu_float_average_xy_frame.get(),
                            view_cache_.get_value<ViewXY>().output_image_accumulation,
                            buffers_.gpu_postprocess_frame_size);

        // XZ view
        if (image_acc_env_.gpu_accumulation_xz_queue && view_cache_.get_value<ViewXZ>().is_image_accumulation_enabled())
            compute_average(image_acc_env_.gpu_accumulation_xz_queue,
                            buffers_.gpu_postprocess_frame_xz.get(),
                            image_acc_env_.gpu_float_average_xz_frame,
                            view_cache_.get_value<ViewXZ>().output_image_accumulation,
                            image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_res());

        // YZ view
        if (image_acc_env_.gpu_accumulation_yz_queue && view_cache_.get_value<ViewYZ>().is_image_accumulation_enabled())
            compute_average(image_acc_env_.gpu_accumulation_yz_queue,
                            buffers_.gpu_postprocess_frame_yz.get(),
                            image_acc_env_.gpu_float_average_yz_frame,
                            view_cache_.get_value<ViewYZ>().output_image_accumulation,
                            image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_res());
    };

    fn_compute_vect_.conditional_push_back(compute_average_lambda);
}

void ImageAccumulation::insert_copy_accumulation_result()
{
    LOG_FUNC();

    auto copy_accumulation_result = [&]()
    {
        // XY view
        if (image_acc_env_.gpu_accumulation_xy_queue && view_cache_.get_value<ViewXY>().is_image_accumulation_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                             image_acc_env_.gpu_float_average_xy_frame,
                             image_acc_env_.gpu_accumulation_xy_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);

        // XZ view
        if (image_acc_env_.gpu_accumulation_xz_queue && view_cache_.get_value<ViewXZ>().is_image_accumulation_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_xz,
                             image_acc_env_.gpu_float_average_xz_frame,
                             image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);

        // YZ view
        if (image_acc_env_.gpu_accumulation_yz_queue && view_cache_.get_value<ViewYZ>().is_image_accumulation_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_yz,
                             image_acc_env_.gpu_float_average_yz_frame,
                             image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);
    };

    fn_compute_vect_.conditional_push_back(copy_accumulation_result);
}
} // namespace holovibes::compute
