#include "image_accumulation.hh"
#include "compute_descriptor.hh"
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
namespace compute
{
ImageAccumulation::ImageAccumulation(FunctionVector& fn_compute_vect,
                                     ImageAccEnv& image_acc_env,
                                     const CoreBuffersEnv& buffers,
                                     const camera::FrameDescriptor& fd,
                                     const holovibes::ComputeDescriptor& cd,
                                     const cudaStream_t& stream)
    : image_acc_env_(image_acc_env)
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , fd_(fd)
    , cd_(cd)
    , stream_(stream)
{
}

void ImageAccumulation::insert_image_accumulation()
{
    insert_compute_average();
    insert_copy_accumulation_result();
}

void ImageAccumulation::allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                                    cuda_tools::UniquePtr<float>& gpu_average_frame,
                                                    const unsigned int accumulation_level,
                                                    const camera::FrameDescriptor fd)
{
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
    // XY view
    if (cd_.get_img_accu_xy_enabled())
    {
        auto new_fd = fd_;
        new_fd.depth = cd_.img_type == ImgType::Composite ? 3 * sizeof(float) : sizeof(float);
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xy_queue,
                                    image_acc_env_.gpu_float_average_xy_frame,
                                    cd_.xy.img_accu_level,
                                    new_fd);
    }

    // XZ view
    if (cd_.get_img_accu_xz_enabled())
    {
        auto new_fd = fd_;
        new_fd.depth = sizeof(float);
        new_fd.height = cd_.time_transformation_size;
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xz_queue,
                                    image_acc_env_.gpu_float_average_xz_frame,
                                    cd_.xz.img_accu_level,
                                    new_fd);
    }

    // YZ view
    if (cd_.get_img_accu_yz_enabled())
    {
        auto new_fd = fd_;
        new_fd.depth = sizeof(float);
        new_fd.width = cd_.time_transformation_size;
        allocate_accumulation_queue(image_acc_env_.gpu_accumulation_yz_queue,
                                    image_acc_env_.gpu_float_average_yz_frame,
                                    cd_.yz.img_accu_level,
                                    new_fd);
    }
}

void ImageAccumulation::dispose()
{
    if (!cd_.get_img_accu_xy_enabled())
        image_acc_env_.gpu_accumulation_xy_queue.reset(nullptr);
    if (!cd_.get_img_accu_xz_enabled())
        image_acc_env_.gpu_accumulation_xz_queue.reset(nullptr);
    if (!cd_.get_img_accu_yz_enabled())
        image_acc_env_.gpu_accumulation_yz_queue.reset(nullptr);
}

void ImageAccumulation::clear()
{
    if (cd_.get_img_accu_xy_enabled())
        image_acc_env_.gpu_accumulation_xy_queue->clear();
    if (cd_.get_img_accu_xz_enabled())
        image_acc_env_.gpu_accumulation_xz_queue->clear();
    if (cd_.get_img_accu_yz_enabled())
        image_acc_env_.gpu_accumulation_yz_queue->clear();
}

void ImageAccumulation::compute_average(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                        float* gpu_input_frame,
                                        float* gpu_ouput_average_frame,
                                        const unsigned int image_acc_level,
                                        const size_t frame_res)
{
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
    auto compute_average_lambda = [&]()
    {
        // XY view
        if (image_acc_env_.gpu_accumulation_xy_queue && cd_.get_img_accu_xy_enabled())
            compute_average(image_acc_env_.gpu_accumulation_xy_queue,
                            buffers_.gpu_postprocess_frame.get(),
                            image_acc_env_.gpu_float_average_xy_frame.get(),
                            cd_.xy.img_accu_level,
                            buffers_.gpu_postprocess_frame_size);

        // XZ view
        if (image_acc_env_.gpu_accumulation_xz_queue && cd_.get_img_accu_xz_enabled())
            compute_average(image_acc_env_.gpu_accumulation_xz_queue,
                            buffers_.gpu_postprocess_frame_xz.get(),
                            image_acc_env_.gpu_float_average_xz_frame,
                            cd_.xz.img_accu_level,
                            image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_res());

        // YZ view
        if (image_acc_env_.gpu_accumulation_yz_queue && cd_.get_img_accu_yz_enabled())
            compute_average(image_acc_env_.gpu_accumulation_yz_queue,
                            buffers_.gpu_postprocess_frame_yz.get(),
                            image_acc_env_.gpu_float_average_yz_frame,
                            cd_.yz.img_accu_level,
                            image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_res());
    };

    fn_compute_vect_.conditional_push_back(compute_average_lambda);
}

void ImageAccumulation::insert_copy_accumulation_result()
{
    auto copy_accumulation_result = [&]()
    {
        // XY view
        if (image_acc_env_.gpu_accumulation_xy_queue && cd_.get_img_accu_xy_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame,
                             image_acc_env_.gpu_float_average_xy_frame,
                             image_acc_env_.gpu_accumulation_xy_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);

        // XZ view
        if (image_acc_env_.gpu_accumulation_xz_queue && cd_.get_img_accu_xz_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_xz,
                             image_acc_env_.gpu_float_average_xz_frame,
                             image_acc_env_.gpu_accumulation_xz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);

        // YZ view
        if (image_acc_env_.gpu_accumulation_yz_queue && cd_.get_img_accu_yz_enabled())
            cudaXMemcpyAsync(buffers_.gpu_postprocess_frame_yz,
                             image_acc_env_.gpu_float_average_yz_frame,
                             image_acc_env_.gpu_accumulation_yz_queue->get_fd().get_frame_size(),
                             cudaMemcpyDeviceToDevice,
                             stream_);
    };

    fn_compute_vect_.conditional_push_back(copy_accumulation_result);
}
} // namespace compute
} // namespace holovibes
