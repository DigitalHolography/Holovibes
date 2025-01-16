#include "image_accumulation.hh"

#include "rect.hh"
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

    insert_copy_accumulation_result(&gpu_postprocess_frame, &gpu_postprocess_frame_xz, &gpu_postprocess_frame_yz);
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
        gpu_average_frame.resize(gpu_accumulation_queue->get_fd().get_frame_size());
    }
}

void ImageAccumulation::init()
{
    LOG_FUNC();

    // XY view
    if (setting<settings::XY>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = static_cast<camera::PixelDepth>(setting<settings::ImageType>() == ImgType::Composite
                                                           ? camera::PixelDepth::Composite
                                                           : camera::PixelDepth::Bits32); // 3 floats or 1 float
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xy_queue,
                                    image_acc_env_.gpu_float_average_xy_frame,
                                    setting<settings::XY>().output_image_accumulation,
                                    new_fd);
    }
}

void ImageAccumulation::init_cuts_queue()
{
    LOG_FUNC();

    // XZ view
    if (setting<settings::XZ>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = camera::PixelDepth::Bits32; // Size of float
        new_fd.height = setting<settings::TimeTransformationSize>();
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xz_queue,
                                    image_acc_env_.gpu_float_average_xz_frame,
                                    setting<settings::XZ>().output_image_accumulation,
                                    new_fd);
    }

    // YZ view
    if (setting<settings::YZ>().output_image_accumulation > 1)
    {
        auto new_fd = fd_;
        new_fd.depth = camera::PixelDepth::Bits32; // Size of float
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
}

void ImageAccumulation::dispose_cuts_queue()
{
    LOG_FUNC();

    if (!(setting<settings::XZ>().output_image_accumulation > 1))
    {
        image_acc_env_.gpu_accumulation_xz_queue.reset(nullptr);
        image_acc_env_.gpu_float_average_xz_frame.reset(nullptr);
    }

    if (!(setting<settings::YZ>().output_image_accumulation > 1))
    {
        image_acc_env_.gpu_accumulation_yz_queue.reset(nullptr);
        image_acc_env_.gpu_float_average_yz_frame.reset(nullptr);
    }
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
        accumulate_images(gpu_ouput_average_frame,
                          static_cast<float*>(gpu_accumulation_queue->get_data()),
                          gpu_accumulation_queue->get_start_index(),
                          gpu_accumulation_queue->get_max_size(),
                          gpu_accumulation_queue->get_size(),
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
        fn_compute_vect_->push_back(
            [&]()
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
        fn_compute_vect_->push_back(
            [&]()
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
        fn_compute_vect_->push_back(
            [&]()
            {
                compute_average(image_acc_env_.gpu_accumulation_yz_queue,
                                buffers_.gpu_postprocess_frame_yz.get(),
                                image_acc_env_.gpu_float_average_yz_frame,
                                setting<settings::YZ>().output_image_accumulation,
                                image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_res());
            });
    }
}

void ImageAccumulation::insert_copy_accumulation_result(float* gpu_postprocess_frame,
                                                        float* gpu_postprocess_frame_xz,
                                                        float* gpu_postprocess_frame_yz)
{
    LOG_FUNC();

    // XY view
    if (image_acc_env_.gpu_accumulation_xy_queue && setting<settings::XY>().output_image_accumulation > 1)
    {
        fn_compute_vect_->push_back(
            [&]()
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
        fn_compute_vect_->push_back(
            [&]()
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
        fn_compute_vect_->push_back(
            [&]()
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
