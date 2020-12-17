/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

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
                   const BatchEnv& batch_env,
                   const TimeTransformationEnv& time_transformation_env,
                   cuda_tools::CufftHandle& plan_unwrap_2d,
                   ComputeDescriptor& cd,
                   const camera::FrameDescriptor& input_fd,
                   const camera::FrameDescriptor& output_fd)
    : pmin_(0)
    , pmax_(0)
    , fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , batch_env_(batch_env)
    , time_transformation_env_(time_transformation_env)
    , unwrap_res_()
    , unwrap_res_2d_()
    , plan_unwrap_2d_(plan_unwrap_2d)
    , cd_(cd)
    , fd_(input_fd)
    , output_fd_(output_fd)
{
}

void Converts::insert_to_float(bool unwrap_2d_requested)
{
    insert_compute_p_accu();
    if (cd_.img_type == ImgType::Composite)
        insert_to_composite();
    else if (cd_.img_type == ImgType::Modulus) // img type in ui : magnitude
        insert_to_modulus();
    else if (cd_.img_type ==
             ImgType::SquaredModulus) // img type in ui : squared magnitude
        insert_to_squaredmodulus();
    else if (cd_.img_type == ImgType::Argument)
        insert_to_argument(unwrap_2d_requested);
    else if (cd_.img_type == ImgType::PhaseIncrease)
        insert_to_phase_increase(unwrap_2d_requested);

    if (cd_.time_transformation == TimeTransformation::PCA &&
        cd_.img_type != ImgType::Composite)
    {
        fn_compute_vect_.conditional_push_back([=]() {
            // Multiply frame by (2 ^ 16) - 1 in case of PCA
            map_multiply(buffers_.gpu_postprocess_frame.get(),
                         buffers_.gpu_postprocess_frame.get(),
                         fd_.frame_res(),
                         static_cast<const float>((2 << 16) - 1));
        });
    }
}

void Converts::insert_to_ushort()
{
    insert_main_ushort();
    if (cd_.time_transformation_cuts_enabled)
        insert_slice_ushort();
}

void Converts::insert_compute_p_accu()
{
    fn_compute_vect_.conditional_push_back([=]() {
        pmin_ = cd_.pindex;
        if (cd_.p_accu_enabled)
            pmax_ = std::max(
                0,
                std::min(pmin_ + cd_.p_acc_level,
                         static_cast<int>(cd_.time_transformation_size)));
        else
            pmax_ = cd_.pindex;
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
                           fd_.frame_res());
    });
}

void Converts::insert_to_squaredmodulus()
{
    fn_compute_vect_.conditional_push_back([=]() {
        complex_to_squared_modulus(buffers_.gpu_postprocess_frame,
                                   time_transformation_env_.gpu_p_acc_buffer,
                                   pmin_,
                                   pmax_,
                                   fd_.frame_res());
    });
}

void Converts::insert_to_composite()
{
    fn_compute_vect_.conditional_push_back([=]() {
        if (!is_between<ushort>(cd_.composite_p_red,
                                0,
                                cd_.time_transformation_size) ||
            !is_between<ushort>(cd_.composite_p_blue,
                                0,
                                cd_.time_transformation_size))
            return;

        if (cd_.composite_kind == CompositeKind::RGB)
            rgb(time_transformation_env_.gpu_p_acc_buffer.get(),
                buffers_.gpu_postprocess_frame,
                fd_.frame_res(),
                cd_.composite_auto_weights_,
                cd_.composite_p_red,
                cd_.composite_p_blue,
                cd_.weight_r,
                cd_.weight_g,
                cd_.weight_b);
        else
            hsv(time_transformation_env_.gpu_p_acc_buffer.get(),
                buffers_.gpu_postprocess_frame,
                fd_.width,
                fd_.height,
                cd_);

        if (cd_.composite_auto_weights_)
            postcolor_normalize(buffers_.gpu_postprocess_frame,
                                fd_.frame_res(),
                                fd_.width,
                                cd_.getCompositeZone(),
                                cd_.weight_r,
                                cd_.weight_g,
                                cd_.weight_b);
    });
}

void Converts::insert_to_argument(bool unwrap_2d_requested)
{
    fn_compute_vect_.conditional_push_back([=]() {
        complex_to_argument(buffers_.gpu_postprocess_frame,
                            time_transformation_env_.gpu_p_acc_buffer,
                            pmin_,
                            pmax_,
                            fd_.frame_res());
    });

    if (unwrap_2d_requested)
    {
        try
        {
            if (!unwrap_res_2d_)
                unwrap_res_2d_.reset(
                    new UnwrappingResources_2d(fd_.frame_res()));
            if (unwrap_res_2d_->image_resolution_ != fd_.frame_res())
                unwrap_res_2d_->reallocate(fd_.frame_res());

            fn_compute_vect_.conditional_push_back([=]() {
                unwrap_2d(buffers_.gpu_postprocess_frame,
                          plan_unwrap_2d_,
                          unwrap_res_2d_.get(),
                          fd_,
                          unwrap_res_2d_->gpu_angle_);
            });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                       buffers_.gpu_postprocess_frame,
                                       unwrap_res_2d_->minmax_buffer_,
                                       fd_.frame_res());
            });
        }
        catch (std::exception& e)
        {
            LOG_ERROR(
                std::string(
                    "Error while trying to convert to float in Argument :") +
                std::string(e.what()));
        }
    }
}

void Converts::insert_to_phase_increase(bool unwrap_2d_requested)
{
    try
    {
        if (!unwrap_res_)
            unwrap_res_.reset(new UnwrappingResources(cd_.unwrap_history_size,
                                                      fd_.frame_res()));
        unwrap_res_->reset(cd_.unwrap_history_size);
        unwrap_res_->reallocate(fd_.frame_res());
        fn_compute_vect_.conditional_push_back([=]() {
            phase_increase(time_transformation_env_.gpu_p_frame,
                           unwrap_res_.get(),
                           fd_.frame_res());
        });

        if (unwrap_2d_requested)
        {
            if (!unwrap_res_2d_)
                unwrap_res_2d_.reset(
                    new UnwrappingResources_2d(fd_.frame_res()));

            if (unwrap_res_2d_->image_resolution_ != fd_.frame_res())
                unwrap_res_2d_->reallocate(fd_.frame_res());

            fn_compute_vect_.conditional_push_back([=]() {
                unwrap_2d(unwrap_res_->gpu_angle_current_,
                          plan_unwrap_2d_,
                          unwrap_res_2d_.get(),
                          fd_,
                          unwrap_res_2d_->gpu_angle_);
            });

            // Converting angle information in floating-point representation.
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float_unwrap2d(unwrap_res_2d_->gpu_angle_,
                                       buffers_.gpu_postprocess_frame,
                                       unwrap_res_2d_->minmax_buffer_,
                                       fd_.frame_res());
            });
        }
        else
            fn_compute_vect_.conditional_push_back([=]() {
                rescale_float(unwrap_res_->gpu_angle_current_,
                              buffers_.gpu_postprocess_frame,
                              fd_.frame_res());
            });
    }
    catch (std::exception& e)
    {
        LOG_ERROR(
            std::string(
                "Error while trying to convert to float in Phase increase :") +
            std::string(e.what()));
    }
}

void Converts::insert_main_ushort()
{
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(buffers_.gpu_postprocess_frame.get(),
                        buffers_.gpu_output_frame.get(),
                        buffers_.gpu_postprocess_frame_size);
    });
}

void Converts::insert_slice_ushort()
{
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(
            buffers_.gpu_postprocess_frame_xz.get(),
            buffers_.gpu_output_frame_xz.get(),
            time_transformation_env_.gpu_output_queue_xz->get_fd().frame_res());
    });
    fn_compute_vect_.conditional_push_back([=]() {
        float_to_ushort(
            buffers_.gpu_postprocess_frame_yz.get(),
            buffers_.gpu_output_frame_yz.get(),
            time_transformation_env_.gpu_output_queue_yz->get_fd().frame_res());
    });
}

void Converts::insert_complex_conversion(Queue& gpu_input_queue)
{
    fn_compute_vect_.push_back([&]() {
        std::lock_guard<std::mutex> m_guard(gpu_input_queue.get_guard());

        // Copy the data from the input queue to the input buffer
        // ALL CALL ARE ASYNCHRONOUS SINCE ALL FFTs AND MEMCPYs ARE CALLED ON
        // STREAM 0
        input_queue_to_input_buffer(
            buffers_.gpu_spatial_transformation_buffer.get(),
            gpu_input_queue.get_data(),
            fd_.frame_res(),
            cd_.batch_size,
            gpu_input_queue.get_start_index(),
            gpu_input_queue.get_max_size(),
            fd_.depth);

        // Dequeue batch size frames
        gpu_input_queue.dequeue_non_mutex(cd_.batch_size);
    });
}
} // namespace compute
} // namespace holovibes
