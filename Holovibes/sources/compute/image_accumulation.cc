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
using cuda_tools::CudaUniquePtr;
using cuda_tools::CufftHandle;
} // namespace holovibes

namespace holovibes::compute
{

void ImageAccumulation::insert_image_accumulation(float& gpu_postprocess_frame,
                                                  unsigned int& gpu_postprocess_frame_size,
                                                  float& gpu_postprocess_frame_xz,
                                                  float& gpu_postprocess_frame_yz)
{
    LOG_FUNC();

    insert_compute_average(gpu_postprocess_frame,
                           gpu_postprocess_frame_size,
                           gpu_postprocess_frame_xz,
                           gpu_postprocess_frame_yz);
    insert_copy_accumulation_result(setting<settings::XY>(),
                                    &gpu_postprocess_frame,
                                    setting<settings::XZ>(),
                                    &gpu_postprocess_frame_xz,
                                    setting<settings::YZ>(),
                                    &gpu_postprocess_frame_yz);
}

void ImageAccumulation::allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                                    cuda_tools::CudaUniquePtr<float>& gpu_average_frame,
                                                    const unsigned int accumulation_level,
                                                    const camera::FrameDescriptor fd)
{
    LOG_FUNC(accumulation_level);

    // If the queue is null or the level has changed
    if (!gpu_accumulation_queue || accumulation_level != gpu_accumulation_queue->get_max_size())
    {
        gpu_accumulation_queue.reset(new Queue(fd, accumulation_level));

        // accumulation queue successfully allocated
        if (!gpu_average_frame)
        {
            auto frame_size = gpu_accumulation_queue->get_fd().get_frame_size();
            gpu_average_frame.resize(frame_size);
        }
    }
}

void ImageAccumulation::init()
{
    LOG_FUNC();

    // XY view
    if (setting<settings::XY>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = setting<settings::ImageType>() == ImgType::Composite ? 3 * sizeof(float) : sizeof(float);
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xy_queue,
                                    image_acc_env_.gpu_float_average_xy_frame,
                                    setting<settings::XY>().output_image_accumulation,
                                    new_fd);
    }

    // XZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::XZ>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = sizeof(float);
        new_fd.height = setting<settings::TimeTransformationSize>();
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xz_queue,
                                    image_acc_env_.gpu_float_average_xz_frame,
                                    setting<settings::XZ>().output_image_accumulation,
                                    new_fd);
    }

    // YZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::YZ>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = sizeof(float);
        new_fd.width = setting<settings::TimeTransformationSize>();
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_yz_queue,
                                    image_acc_env_.gpu_float_average_yz_frame,
                                    setting<settings::YZ>().output_image_accumulation,
                                    new_fd);
    }
}

void ImageAccumulation::dispose()
{
    LOG_FUNC();

    if (!(setting<settings::XY>().output_image_accumulation > 1))
        image_acc_env_.gpu_accumulation_xy_queue.reset(nullptr);
    if (setting<settings::CutsViewEnabled>() && !(setting<settings::XZ>().output_image_accumulation > 1))
        image_acc_env_.gpu_accumulation_xz_queue.reset(nullptr);
    if (setting<settings::CutsViewEnabled>() && !(setting<settings::YZ>().output_image_accumulation > 1))
        image_acc_env_.gpu_accumulation_yz_queue.reset(nullptr);
}

void ImageAccumulation::clear()
{
    LOG_FUNC();

    if (setting<settings::XY>().output_image_accumulation > 1)
        image_acc_env_.gpu_accumulation_xy_queue->clear();
    if (setting<settings::CutsViewEnabled>() && setting<settings::XZ>().output_image_accumulation > 1)
        image_acc_env_.gpu_accumulation_xz_queue->clear();
    if (setting<settings::CutsViewEnabled>() && setting<settings::YZ>().output_image_accumulation > 1)
        image_acc_env_.gpu_accumulation_yz_queue->clear();
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

void ImageAccumulation::insert_compute_average(float& gpu_postprocess_frame,
                                               unsigned int& gpu_postprocess_frame_size,
                                               float& gpu_postprocess_frame_xz,
                                               float& gpu_postprocess_frame_yz)
{
    LOG_FUNC();

    // XY view
    if (image_acc_env_.gpu_accumulation_xy_queue && setting<settings::XY>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            compute_average(image_acc_env_.gpu_accumulation_xy_queue,
                        &gpu_postprocess_frame,
                        image_acc_env_.gpu_float_average_xy_frame.get(),
                        setting<settings::XY>().output_image_accumulation,
                        buffers_.gpu_postprocess_frame_size);
        });
    }

    // XZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::XZ>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            compute_average(image_acc_env_.gpu_accumulation_xz_queue,
                        buffers_.gpu_postprocess_frame_xz.get(),
                        image_acc_env_.gpu_float_average_xz_frame,
                        setting<settings::XZ>().output_image_accumulation,
                        image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_res());
        });
    }
        


    // YZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::YZ>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            compute_average(image_acc_env_.gpu_accumulation_yz_queue,
                        buffers_.gpu_postprocess_frame_yz.get(),
                        image_acc_env_.gpu_float_average_yz_frame,
                        setting<settings::YZ>().output_image_accumulation,
                        image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_res());
        });
    }
}

void ImageAccumulation::insert_copy_accumulation_result(const holovibes::ViewXYZ& const_view_xy,
                                                        float* gpu_postprocess_frame,
                                                        const holovibes::ViewXYZ& const_view_xz,
                                                        float* gpu_postprocess_frame_xz,
                                                        const holovibes::ViewXYZ& const_view_yz,
                                                        float* gpu_postprocess_frame_yz)
{
    LOG_FUNC();

    // XY view
    if (image_acc_env_.gpu_accumulation_xy_queue && setting<settings::XY>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                             image_acc_env_.gpu_float_average_xy_frame,
                             image_acc_env_.gpu_accumulation_xy_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
    }

    // XZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::XZ>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_xz,
                             image_acc_env_.gpu_float_average_xz_frame,
                             image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
    }
            
    // YZ view
    if (setting<settings::CutsViewEnabled>() && setting<settings::YZ>().output_image_accumulation > 1)
    {
        fn_compute_vect_.conditional_push_back([&]()
        {
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_yz,
                             image_acc_env_.gpu_float_average_yz_frame,
                             image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);
        });
    }
}
} // namespace holovibes::compute
