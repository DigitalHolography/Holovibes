#include "converts.hh"
#include "frame_desc.hh"

#include "icompute.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "tools_conversion.cuh"
#include "composite.cuh"
#include "rgb.cuh"
#include "hsv.cuh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "tools_unwrap.cuh"
#include "map.cuh"
#include "API.hh"

#include <mutex>

namespace holovibes::compute
{

void Converts::insert_to_float(bool unwrap_2d_requested,
                               float* buffers_gpu_postprocess_frame)
{
    LOG_FUNC(unwrap_2d_requested);
    ImgType img_type = setting<settings::ImageType>();
    insert_compute_p_accu();
    if (img_type == ImgType::Composite)
        insert_to_composite(buffers_gpu_postprocess_frame);
    else if (img_type == ImgType::Modulus) // img type in ui : magnitude
        insert_to_modulus(buffers_gpu_postprocess_frame);
    else if (img_type == ImgType::SquaredModulus) // img type in ui : squared magnitude
        insert_to_squaredmodulus(buffers_gpu_postprocess_frame);
    else if (img_type == ImgType::Argument)
        insert_to_argument(unwrap_2d_requested, buffers_gpu_postprocess_frame);
    else if (img_type == ImgType::PhaseIncrease)
        insert_to_phase_increase(unwrap_2d_requested, buffers_gpu_postprocess_frame);

    if (setting<settings::TimeTransformation>() == TimeTransformation::PCA && img_type != ImgType::Composite)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                // Multiply frame by (2 ^ 16) - 1 in case of PCA
                map_multiply(buffers_gpu_postprocess_frame,
                             buffers_gpu_postprocess_frame,
                             fd_.get_frame_res(),
                             static_cast<const float>((2 << 16) - 1),
                             stream_);
            });
    }
}

void Converts::insert_to_ushort()
{
    LOG_FUNC();

    insert_main_ushort();
    if (setting<settings::CutsViewEnabled>())
        insert_slice_ushort();
    if (setting<settings::Filter2dViewEnabled>())
        insert_filter2d_ushort();
}

void Converts::insert_compute_p_accu()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            auto p = setting<settings::P>();
            pmin_ = p.start;
            if (p.width != 0)
                pmax_ = std::max(0, std::min<int>(pmin_ + p.width, static_cast<int>(setting<settings::TimeTransformationSize>())));
            else
                pmax_ = p.start;
        });
}

// we use gpu_input_buffer because when time_transformation_size = 1,
// gpu_p_acc_buffer is not used.
void Converts::insert_to_modulus(float* gpu_postprocess_frame)
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            complex_to_modulus(gpu_postprocess_frame,
                               time_transformation_env_.gpu_p_acc_buffer,
                               pmin_,
                               pmax_,
                               fd_.get_frame_res(),
                               stream_);
        });
}

void Converts::insert_to_squaredmodulus(float* gpu_postprocess_frame)
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            complex_to_squared_modulus(gpu_postprocess_frame,
                                       time_transformation_env_.gpu_p_acc_buffer,
                                       pmin_,
                                       pmax_,
                                       fd_.get_frame_res(),
                                       stream_);
        });
}

void Converts::insert_to_composite(float* gpu_postprocess_frame)
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            CompositeRGB rgb_struct = setting<settings::RGB>();
            auto time_transformation_size = setting<settings::TimeTransformationSize>();
            if (!is_between<ushort>(rgb_struct.frame_index.min, 0, time_transformation_size) ||
                !is_between<ushort>(rgb_struct.frame_index.max, 0, time_transformation_size))
                return;

            if (setting<settings::CompositeKind>() == CompositeKind::RGB)
                rgb(time_transformation_env_.gpu_p_acc_buffer.get(),
                    gpu_postprocess_frame,
                    fd_.get_frame_res(),
                    setting<settings::CompositeAutoWeights>(),
                    rgb_struct.frame_index.min,
                    rgb_struct.frame_index.max,
                    rgb_struct.weight,
                    stream_);
            else
                hsv(time_transformation_env_.gpu_p_acc_buffer.get(),
                    gpu_postprocess_frame,
                    fd_.width,
                    fd_.height,
                    stream_,
                    time_transformation_size,
                    setting<settings::HSV>());

            if (setting<settings::CompositeAutoWeights>())
            {
                const uchar pixel_depth = 3;
                const int factor = 10;
                float* averages = new float[pixel_depth];
                postcolor_normalize(gpu_postprocess_frame,
                                    fd_.height,
                                    fd_.width,
                                    setting<settings::CompositeZone>(),
                                    pixel_depth,
                                    averages,
                                    stream_);
                if (pixel_depth >= 3)
                {
                    double max = std::max(std::max(averages[0], averages[1]), averages[2]);
                    //!!!!! replace api by update_setting
                    api::set_weight_rgb((static_cast<double>(averages[0]) / max) * factor,
                                                   (static_cast<double>(averages[1]) / max) * factor,
                                                   (static_cast<double>(averages[2]) / max) * factor);
                }
            }
        });
}

void Converts::insert_to_argument(bool unwrap_2d_requested, float* gpu_postprocess_frame)
{
    LOG_FUNC(unwrap_2d_requested);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            complex_to_argument(gpu_postprocess_frame,
                                time_transformation_env_.gpu_p_acc_buffer,
                                pmin_,
                                pmax_,
                                fd_.get_frame_res(),
                                stream_);
        });

    if (unwrap_2d_requested)
    {
        try
        {
            if (!unwrap_res_2d_)
                unwrap_res_2d_.reset(new UnwrappingResources_2d(fd_.get_frame_res(), stream_));
            if (unwrap_res_2d_->image_resolution_ != fd_.get_frame_res())
                unwrap_res_2d_->reallocate(fd_.get_frame_res());

            fn_compute_vect_.conditional_push_back(
                [=]() {
                    unwrap_2d(gpu_postprocess_frame,
                              plan_unwrap_2d_,
                              unwrap_res_2d_.get(),
                              fd_,
                              unwrap_res_2d_->gpu_angle_,
                              stream_);
                });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back(
                [=]()
                {
                    rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                           gpu_postprocess_frame,
                                           unwrap_res_2d_->minmax_buffer_,
                                           fd_.get_frame_res(),
                                           stream_);
                });
        }
        catch (std::exception& e)
        {
            LOG_ERROR("Error while trying to convert to float in Argument : {}", e.what());
        }
    }
}

void Converts::insert_to_phase_increase(bool unwrap_2d_requested,
                                        float* gpu_postprocess_frame)
{
    LOG_FUNC(unwrap_2d_requested);

    try
    {
        if (!unwrap_res_)
            unwrap_res_.reset(new UnwrappingResources(setting<settings::UnwrapHistorySize>(), fd_.get_frame_res(), stream_));
        unwrap_res_->reset(setting<settings::UnwrapHistorySize>());
        unwrap_res_->reallocate(fd_.get_frame_res());
        fn_compute_vect_.conditional_push_back(
            [=]()
            { phase_increase(time_transformation_env_.gpu_p_frame, unwrap_res_.get(), fd_.get_frame_res(), stream_); });

        if (unwrap_2d_requested)
        {
            if (!unwrap_res_2d_)
                unwrap_res_2d_.reset(new UnwrappingResources_2d(fd_.get_frame_res(), stream_));

            if (unwrap_res_2d_->image_resolution_ != fd_.get_frame_res())
                unwrap_res_2d_->reallocate(fd_.get_frame_res());

            fn_compute_vect_.conditional_push_back(
                [=]()
                {
                    unwrap_2d(unwrap_res_->gpu_angle_current_,
                              plan_unwrap_2d_,
                              unwrap_res_2d_.get(),
                              fd_,
                              unwrap_res_2d_->gpu_angle_,
                              stream_);
                });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back(
                [=]()
                {
                    rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                           gpu_postprocess_frame,
                                           unwrap_res_2d_->minmax_buffer_,
                                           fd_.get_frame_res(),
                                           stream_);
                });
        }
        else
            fn_compute_vect_.conditional_push_back(
                [=]() {
                    rescale_float(unwrap_res_->gpu_angle_current_,
                                  gpu_postprocess_frame,
                                  fd_.get_frame_res(),
                                  stream_);
                });
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Error while trying to convert to float in Phase increase : {}", e.what());
    }
}

void Converts::insert_main_ushort()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            float_to_ushort(buffers_.gpu_postprocess_frame.get(),
                            buffers_.gpu_output_frame.get(),
                            buffers_.gpu_postprocess_frame_size,
                            stream_);
        });
}

void Converts::insert_slice_ushort()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            float_to_ushort(buffers_.gpu_postprocess_frame_xz.get(),
                            buffers_.gpu_output_frame_xz.get(),
                            time_transformation_env_.gpu_output_queue_xz->get_fd().get_frame_res(),
                            stream_);
        });
    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            float_to_ushort(buffers_.gpu_postprocess_frame_yz.get(),
                            buffers_.gpu_output_frame_yz.get(),
                            time_transformation_env_.gpu_output_queue_yz->get_fd().get_frame_res(),
                            stream_);
        });
}

void Converts::insert_filter2d_ushort()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            float_to_ushort_normalized(buffers_.gpu_float_filter2d_frame.get(),
                                       buffers_.gpu_filter2d_frame.get(),
                                       buffers_.gpu_postprocess_frame_size,
                                       stream_);
        });
}

void Converts::insert_complex_conversion(BatchInputQueue& gpu_input_queue)
{
    LOG_FUNC(fd_.depth);
    fn_compute_vect_.push_back(
        [&]()
        {
            static const BatchInputQueue::dequeue_func_t convert_to_complex = [](const void* const src,
                                                                                 void* const dest,
                                                                                 const uint batch_size,
                                                                                 const size_t frame_res,
                                                                                 const uint depth,
                                                                                 const cudaStream_t stream)
            { input_queue_to_input_buffer(dest, src, frame_res, batch_size, depth, stream); };

            void* output = buffers_.gpu_spatial_transformation_buffer.get();
            gpu_input_queue.dequeue(output, fd_.depth, convert_to_complex);
        });
}
} // namespace holovibes::compute
