#include "converts.hh"
#include "frame_desc.hh"
#include "compute_descriptor.hh"
#include "icompute.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "tools_conversion.cuh"
#include "composite.cuh"
#include "hsv.cuh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "tools_unwrap.cuh"
#include "map.cuh"

#include <mutex>

namespace holovibes
{
namespace compute
{
Converts::Converts(FunctionVector& fn_compute_vect,
                   const CoreBuffersEnv& buffers,
                   const TimeTransformationEnv& time_transformation_env,
                   cuda_tools::CufftHandle& plan_unwrap_2d,
                   ComputeDescriptor& cd,
                   const camera::FrameDescriptor& input_fd,
                   const cudaStream_t& stream,
                   ComputeCache::Cache& compute_cache,
                   CompositeCache::Cache& composite_cache,
                   ViewCache::Cache& view_cache,
                   ZoneCache::Cache& zone_cache)
    : pmin_(0)
    , pmax_(0)
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , time_transformation_env_(time_transformation_env)
    , plan_unwrap_2d_(plan_unwrap_2d)
    , fd_(input_fd)
    , cd_(cd)
    , stream_(stream)
    , compute_cache_(compute_cache)
    , composite_cache_(composite_cache)
    , view_cache_(view_cache)
    , zone_cache_(zone_cache)
{
}

void Converts::insert_to_float(bool unwrap_2d_requested)
{
    insert_compute_p_accu();
    if (view_cache_.get_img_type() == ImgType::Composite)
        insert_to_composite();
    else if (view_cache_.get_img_type() == ImgType::Modulus) // img type in ui : magnitude
        insert_to_modulus();
    else if (view_cache_.get_img_type() == ImgType::SquaredModulus) // img type in ui : squared magnitude
        insert_to_squaredmodulus();
    else if (view_cache_.get_img_type() == ImgType::Argument)
        insert_to_argument(unwrap_2d_requested);
    else if (view_cache_.get_img_type() == ImgType::PhaseIncrease)
        insert_to_phase_increase(unwrap_2d_requested);

    if (compute_cache_.get_time_transformation() == TimeTransformation::PCA &&
        view_cache_.get_img_type() != ImgType::Composite)
    {
        fn_compute_vect_.conditional_push_back([=]() {
            // Multiply frame by (2 ^ 16) - 1 in case of PCA
            map_multiply(buffers_.gpu_postprocess_frame.get(),
                         buffers_.gpu_postprocess_frame.get(),
                         fd_.get_frame_res(),
                         static_cast<const float>((2 << 16) - 1),
                         stream_);
        });
    }
}

void Converts::insert_to_ushort()
{
    insert_main_ushort();
    if (view_cache_.get_cuts_view_enabled())
        insert_slice_ushort();
    if (view_cache_.get_filter2d_view_enabled())
        insert_filter2d_ushort();
}

void Converts::insert_compute_p_accu()
{
    fn_compute_vect_.conditional_push_back([=]() {
        View_PQ p = view_cache_.get_p();
        pmin_ = p.index;
        if (p.accu_level != 0)
            pmax_ = std::max(
                0,
                std::min<int>(pmin_ + p.accu_level, static_cast<int>(compute_cache_.get_time_transformation_size())));
        else
            pmax_ = p.index;
    });
}

// we use gpu_input_buffer because when time_transformation_size = 1,
// gpu_p_acc_buffer is not used.
void Converts::insert_to_modulus()
{
    fn_compute_vect_.conditional_push_back([=]() {
        complex_to_modulus(buffers_.gpu_postprocess_frame,
                           time_transformation_env_.gpu_p_acc_buffer,
                           pmin_,
                           pmax_,
                           fd_.get_frame_res(),
                           stream_);
    });
}

void Converts::insert_to_squaredmodulus()
{
    fn_compute_vect_.conditional_push_back([=]() {
        complex_to_squared_modulus(buffers_.gpu_postprocess_frame,
                                   time_transformation_env_.gpu_p_acc_buffer,
                                   pmin_,
                                   pmax_,
                                   fd_.get_frame_res(),
                                   stream_);
    });
}

void Converts::insert_to_composite()
{
    fn_compute_vect_.conditional_push_back([=]() {
        Composite_RGB rgb_struct = composite_cache_.get_rgb();
        if (!is_between<ushort>(rgb_struct.p_min, 0, compute_cache_.get_time_transformation_size()) ||
            !is_between<ushort>(rgb_struct.p_max, 0, compute_cache_.get_time_transformation_size()))
            return;

        if (composite_cache_.get_composite_kind() == CompositeKind::RGB)
            rgb(time_transformation_env_.gpu_p_acc_buffer.get(),
                buffers_.gpu_postprocess_frame,
                fd_.get_frame_res(),
                composite_cache_.get_composite_auto_weights(),
                rgb_struct.p_min,
                rgb_struct.p_max,
                rgb_struct.weight_r,
                rgb_struct.weight_g,
                rgb_struct.weight_b,
                stream_);
        else
            hsv(time_transformation_env_.gpu_p_acc_buffer.get(),
                buffers_.gpu_postprocess_frame,
                fd_.width,
                fd_.height,
                cd_,
                stream_,
                compute_cache_.get_time_transformation_size(),
                composite_cache_.get_hsv_const_ref());

        if (composite_cache_.get_composite_auto_weights())
            postcolor_normalize(buffers_.gpu_postprocess_frame,
                                fd_.get_frame_res(),
                                fd_.width,
                                zone_cache_.get_composite_zone(),
                                rgb_struct.weight_r,
                                rgb_struct.weight_g,
                                rgb_struct.weight_b,
                                stream_);
    });
}

void Converts::insert_to_argument(bool unwrap_2d_requested)
{
    fn_compute_vect_.conditional_push_back([=]() {
        complex_to_argument(buffers_.gpu_postprocess_frame,
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

            fn_compute_vect_.conditional_push_back([=]() {
                unwrap_2d(buffers_.gpu_postprocess_frame,
                          plan_unwrap_2d_,
                          unwrap_res_2d_.get(),
                          fd_,
                          unwrap_res_2d_->gpu_angle_,
                          stream_);
            });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                       buffers_.gpu_postprocess_frame,
                                       unwrap_res_2d_->minmax_buffer_,
                                       fd_.get_frame_res(),
                                       stream_);
            });
        }
        catch (std::exception& e)
        {
            LOG_ERROR << "Error while trying to convert to float in Argument :" << e.what();
        }
    }
}

void Converts::insert_to_phase_increase(bool unwrap_2d_requested)
{
    try
    {
        if (!unwrap_res_)
            unwrap_res_.reset(
                new UnwrappingResources(compute_cache_.get_unwrap_history_size(), fd_.get_frame_res(), stream_));
        unwrap_res_->reset(compute_cache_.get_unwrap_history_size());
        unwrap_res_->reallocate(fd_.get_frame_res());
        fn_compute_vect_.conditional_push_back([=]() {
            phase_increase(time_transformation_env_.gpu_p_frame, unwrap_res_.get(), fd_.get_frame_res(), stream_);
        });

        if (unwrap_2d_requested)
        {
            if (!unwrap_res_2d_)
                unwrap_res_2d_.reset(new UnwrappingResources_2d(fd_.get_frame_res(), stream_));

            if (unwrap_res_2d_->image_resolution_ != fd_.get_frame_res())
                unwrap_res_2d_->reallocate(fd_.get_frame_res());

            fn_compute_vect_.conditional_push_back([=]() {
                unwrap_2d(unwrap_res_->gpu_angle_current_,
                          plan_unwrap_2d_,
                          unwrap_res_2d_.get(),
                          fd_,
                          unwrap_res_2d_->gpu_angle_,
                          stream_);
            });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                       buffers_.gpu_postprocess_frame,
                                       unwrap_res_2d_->minmax_buffer_,
                                       fd_.get_frame_res(),
                                       stream_);
            });
        }
        else
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float(unwrap_res_->gpu_angle_current_,
                              buffers_.gpu_postprocess_frame,
                              fd_.get_frame_res(),
                              stream_);
            });
    }
    catch (std::exception& e)
    {
        LOG_ERROR << "Error while trying to convert to float in Phase increase: " << e.what();
    }
}

void Converts::insert_main_ushort()
{
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(buffers_.gpu_postprocess_frame.get(),
                        buffers_.gpu_output_frame.get(),
                        buffers_.gpu_postprocess_frame_size,
                        stream_);
    });
}

void Converts::insert_slice_ushort()
{
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(buffers_.gpu_postprocess_frame_xz.get(),
                        buffers_.gpu_output_frame_xz.get(),
                        time_transformation_env_.gpu_output_queue_xz->get_fd().get_frame_res(),
                        stream_);
    });
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(buffers_.gpu_postprocess_frame_yz.get(),
                        buffers_.gpu_output_frame_yz.get(),
                        time_transformation_env_.gpu_output_queue_yz->get_fd().get_frame_res(),
                        stream_);
    });
}

void Converts::insert_filter2d_ushort()
{
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(buffers_.gpu_float_filter2d_frame.get(),
                        buffers_.gpu_filter2d_frame.get(),
                        buffers_.gpu_postprocess_frame_size,
                        stream_);
    });
}

void Converts::insert_complex_conversion(BatchInputQueue& gpu_input_queue)
{
    fn_compute_vect_.push_back([&]() {
        static const BatchInputQueue::dequeue_func_t convert_to_complex = [](const void* const src,
                                                                             void* const dest,
                                                                             const uint batch_size,
                                                                             const size_t frame_res,
                                                                             const uint depth,
                                                                             const cudaStream_t stream) {
            input_queue_to_input_buffer(dest, src, frame_res, batch_size, depth, stream);
        };

        void* output = buffers_.gpu_spatial_transformation_buffer.get();

        gpu_input_queue.dequeue(output, fd_.depth, convert_to_complex);
    });
}
} // namespace compute
} // namespace holovibes
