#include "rendering.hh"
#include "frame_desc.hh"
#include "icompute.hh"

#include "concurrent_deque.hh"
#include "contrast_correction.cuh"
#include "chart.cuh"
#include "stft.cuh"
#include "percentile.cuh"
#include "map.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "API.hh"

namespace holovibes::compute
{
Rendering::Rendering(FunctionVector& fn_compute_vect,
                     const CoreBuffersEnv& buffers,
                     ChartEnv& chart_env,
                     const ImageAccEnv& image_acc_env,
                     const TimeTransformationEnv& time_transformation_env,
                     const FrameDescriptor& input_fd,
                     const FrameDescriptor& output_fd,
                     const cudaStream_t& stream,
                     PipeAdvancedCache& advanced_cache,
                     PipeComputeCache& compute_cache,
                     PipeExportCache& export_cache,
                     PipeViewCache& view_cache,
                     PipeZoneCache& zone_cache)
    : fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , chart_env_(chart_env)
    , time_transformation_env_(time_transformation_env)
    , image_acc_env_(image_acc_env)
    , input_fd_(input_fd)
    , fd_(output_fd)
    , stream_(stream)
    , advanced_cache_(advanced_cache)
    , compute_cache_(compute_cache)
    , export_cache_(export_cache)
    , view_cache_(view_cache)
    , zone_cache_(zone_cache)
{
    // Hold 2 float values (min and max)
    cudaXMallocHost(&percent_min_max_, 2 * sizeof(float));
}

Rendering::~Rendering() { cudaXFreeHost(percent_min_max_); }

void Rendering::insert_fft_shift()
{
    LOG_FUNC();

    if (view_cache_.get_value<FftShiftEnabled>())
    {
        if (compute_cache_.get_value<ImageType>() == ImageTypeEnum::Composite)
            fn_compute_vect_.conditional_push_back(
                [=]() {
                    shift_corners(reinterpret_cast<float3*>(buffers_.gpu_postprocess_frame.get()),
                                  1,
                                  fd_.width,
                                  fd_.height,
                                  stream_);
                });
        else
            fn_compute_vect_.conditional_push_back(
                [=]() { shift_corners(buffers_.gpu_postprocess_frame, 1, fd_.width, fd_.height, stream_); });
    }
}

void Rendering::insert_chart()
{
    LOG_FUNC();

    if (export_cache_.get_value<Record>().is_running)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                auto signal_zone = zone_cache_.get_value<SignalZone>();
                auto noise_zone = zone_cache_.get_value<NoiseZone>();

                if (signal_zone.width() == 0 || signal_zone.height() == 0 || noise_zone.width() == 0 ||
                    noise_zone.height() == 0)
                    return;

                ChartPoint point = make_chart_plot(buffers_.gpu_postprocess_frame,
                                                   input_fd_.width,
                                                   input_fd_.height,
                                                   signal_zone,
                                                   noise_zone,
                                                   stream_);

                if (view_cache_.get_value<ChartDisplayEnabled>())
                    chart_env_.chart_display_queue_->push_back(point);

                if (export_cache_.get_value<Record>().is_running && chart_env_.current_nb_points_recorded > 0)
                {
                    chart_env_.chart_record_queue_->push_back(point);
                    --chart_env_.current_nb_points_recorded;
                }
            });
    }
}

void Rendering::insert_log()
{
    LOG_FUNC();

    if (view_cache_.get_value<ViewXY>().log_enabled)
        insert_main_log();
    if (view_cache_.get_value<CutsViewEnabled>())
        insert_slice_log();
    if (view_cache_.get_value<ViewFilter2D>().log_enabled)
        insert_filter2d_view_log();
}

void Rendering::insert_main_log()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            map_log10(buffers_.gpu_postprocess_frame.get(),
                      buffers_.gpu_postprocess_frame.get(),
                      buffers_.gpu_postprocess_frame_size,
                      stream_);
        });
}
void Rendering::insert_slice_log()
{
    LOG_FUNC();

    if (view_cache_.get_value<ViewXZ>().log_enabled)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                map_log10(buffers_.gpu_postprocess_frame_xz.get(),
                          buffers_.gpu_postprocess_frame_xz.get(),
                          fd_.width * compute_cache_.get_value<TimeTransformationSize>(),
                          stream_);
            });
    }
    if (view_cache_.get_value<ViewYZ>().log_enabled)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                map_log10(buffers_.gpu_postprocess_frame_yz.get(),
                          buffers_.gpu_postprocess_frame_yz.get(),
                          fd_.height * compute_cache_.get_value<TimeTransformationSize>(),
                          stream_);
            });
    }
}

void Rendering::insert_filter2d_view_log()
{
    LOG_FUNC();

    if (api::detail::get_value<Filter2DViewEnabled>())
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                map_log10(buffers_.gpu_float_filter2d_frame.get(),
                          buffers_.gpu_float_filter2d_frame.get(),
                          fd_.width * fd_.height,
                          stream_);
            });
    }
}

template <WindowKind View>
void Rendering::insert_apply_contrast()
{
    LOG_FUNC();

    fn_compute_vect_.conditional_push_back(
        [&]()
        {
            float* input = nullptr;
            uint size = 0;
            ViewWindow wind;

            // FIXME API : View should be class inherit to remove thoses if
            if constexpr (View == WindowKind::ViewXY)
            {
                if (api::get_view_xy().contrast.enabled == false)
                    return;

                input = buffers_.gpu_postprocess_frame;
                size = buffers_.gpu_postprocess_frame_size;
                wind = view_cache_.get_value<ViewXY>();
            }

            else if constexpr (View == WindowKind::ViewXZ)
            {
                if (api::get_cuts_view_enabled() == false)
                    return;
                if (api::get_view_xz().contrast.enabled == false)
                    return;

                input = buffers_.gpu_postprocess_frame_xz.get();
                size = fd_.width * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewXZ>();
            }

            else if constexpr (View == WindowKind::ViewYZ)
            {
                if (api::get_cuts_view_enabled() == false)
                    return;
                if (api::get_view_yz().contrast.enabled == false)
                    return;

                input = buffers_.gpu_postprocess_frame_yz.get();
                size = fd_.height * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewYZ>();
            }

            else if constexpr (View == WindowKind::ViewFilter2D)
            {
                if (api::get_filter2d_view_enabled() == false)
                    return;
                if (api::get_view_filter2d().contrast.enabled == false)
                    return;

                input = buffers_.gpu_float_filter2d_frame.get();
                size = fd_.width * fd_.height;
                wind = view_cache_.get_value<ViewFilter2D>();
            }

            constexpr ushort dynamic_range = 65535;
            float min = wind.contrast.min;
            float max = wind.contrast.max;

            if (wind.contrast.invert)
            {
                min = wind.contrast.max;
                max = wind.contrast.min;
            }

            apply_contrast_correction(input, size, dynamic_range, min, max, stream_);
        });
}

void Rendering::insert_contrast()
{
    LOG_FUNC();

    // FIXME API : All thoses function should be a unique job/lambda
    insert_request_exec_contrast();
    insert_apply_contrast<WindowKind::ViewXY>();
    insert_apply_contrast<WindowKind::ViewXZ>();
    insert_apply_contrast<WindowKind::ViewYZ>();
    insert_apply_contrast<WindowKind::ViewFilter2D>();
}

void Rendering::insert_request_exec_contrast()
{
    LOG_FUNC();

    // requested check are inside the lambda so that we don't need to
    // refresh the pipe at each autocontrast
    auto lambda = [&]()
    {
        // Compute autocontrast once the gpu time transformation queue is full
        if (!time_transformation_env_.gpu_time_transformation_queue->is_full())
            return;

        if (api::get_view_xy().contrast.enabled)
            if (api::get_view_xy().contrast.auto_refresh || has_requested_view_exec_contrast(WindowKind::ViewXY))
            {
                reset_view_exec_contrast(WindowKind::ViewXY);
                autocontrast_caller(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, 0, WindowKind::ViewXY);
            }

        if (api::get_cuts_view_enabled())
        {
            if (api::get_view_xz().contrast.enabled)
            {
                if (api::get_view_xz().contrast.auto_refresh || has_requested_view_exec_contrast(WindowKind::ViewXZ))
                {
                    reset_view_exec_contrast(WindowKind::ViewXZ);
                    autocontrast_caller(buffers_.gpu_postprocess_frame_xz.get(),
                                        fd_.width,
                                        compute_cache_.get_value<TimeTransformationSize>(),
                                        advanced_cache_.get_value<ContrastThreshold>().frame_index_offset,
                                        WindowKind::ViewXZ);
                }
            }

            if (api::get_view_yz().contrast.enabled)
            {
                if (api::get_view_yz().contrast.auto_refresh || has_requested_view_exec_contrast(WindowKind::ViewYZ))
                {
                    reset_view_exec_contrast(WindowKind::ViewYZ);
                    autocontrast_caller(buffers_.gpu_postprocess_frame_yz.get(),
                                        compute_cache_.get_value<TimeTransformationSize>(),
                                        fd_.height,
                                        advanced_cache_.get_value<ContrastThreshold>().frame_index_offset,
                                        WindowKind::ViewYZ);
                }
            }
        }

        if (api::get_filter2d_view_enabled() && api::get_view_filter2d().contrast.enabled)
            if (api::get_view_filter2d().contrast.auto_refresh ||
                has_requested_view_exec_contrast(WindowKind::ViewFilter2D))
            {
                reset_view_exec_contrast(WindowKind::ViewFilter2D);
                autocontrast_caller(buffers_.gpu_float_filter2d_frame.get(),
                                    fd_.width,
                                    fd_.height,
                                    0,
                                    WindowKind::ViewFilter2D);
            }
    };

    fn_compute_vect_.conditional_push_back(lambda);
}

void Rendering::insert_clear_image_accumulation()
{
    LOG_FUNC();

    auto lambda_clear_image_accumulation = [&]()
    {
        if (has_requested_view_clear_image_accumulation(WindowKind::ViewXY))
        {
            reset_view_clear_image_accumulation(WindowKind::ViewXY);
            api::get_compute_pipe().get_image_acc_env().gpu_accumulation_xy_queue->clear();
        }

        if (has_requested_view_clear_image_accumulation(WindowKind::ViewXZ))
        {
            reset_view_clear_image_accumulation(WindowKind::ViewXZ);
            api::get_compute_pipe().get_image_acc_env().gpu_accumulation_xz_queue->clear();
        }

        if (has_requested_view_clear_image_accumulation(WindowKind::ViewYZ))
        {
            reset_view_clear_image_accumulation(WindowKind::ViewYZ);
            api::get_compute_pipe().get_image_acc_env().gpu_accumulation_yz_queue->clear();
        }
    };

    fn_compute_vect_.conditional_push_back(lambda_clear_image_accumulation);
}

void Rendering::autocontrast_caller(
    float* input, const uint width, const uint height, const uint offset, WindowKind view)
{
    constexpr uint percent_size = 2;

    const float percent_in[percent_size] = {advanced_cache_.get_value<ContrastThreshold>().lower,
                                            advanced_cache_.get_value<ContrastThreshold>().upper};
    switch (view)
    {
    case WindowKind::ViewXY:
    case WindowKind::ViewXZ:
    case WindowKind::ViewFilter2D:
        // No offset
        compute_percentile_xy_view(
            input,
            width,
            height,
            (view == WindowKind::ViewXY) ? 0 : offset,
            percent_in,
            percent_min_max_,
            percent_size,
            zone_cache_.get_value<ReticleZone>(),
            (view == WindowKind::ViewFilter2D) ? false : view_cache_.get_value<Reticle>().display_enabled,
            stream_);
        break;
    case WindowKind::ViewYZ: // TODO: finished refactoring to remove this switch
        compute_percentile_yz_view(input,
                                   width,
                                   height,
                                   offset,
                                   percent_in,
                                   percent_min_max_,
                                   percent_size,
                                   zone_cache_.get_value<ReticleZone>(),
                                   view_cache_.get_value<Reticle>().display_enabled,
                                   stream_);
        break;
    }

    // FIXME API : deso but view are too weirdy coded
    switch (view)
    {
    case WindowKind::ViewXY:
        GSH::instance().get_view_cache().get_value_ref_W<ViewXY>().contrast.min = percent_min_max_[0];
        GSH::instance().get_view_cache().get_value_ref_W<ViewXY>().contrast.max = percent_min_max_[1];
        break;
    case WindowKind::ViewXZ:
        GSH::instance().get_view_cache().get_value_ref_W<ViewXZ>().contrast.min = percent_min_max_[0];
        GSH::instance().get_view_cache().get_value_ref_W<ViewXZ>().contrast.max = percent_min_max_[1];
        break;
    case WindowKind::ViewYZ:
        GSH::instance().get_view_cache().get_value_ref_W<ViewYZ>().contrast.min = percent_min_max_[0];
        GSH::instance().get_view_cache().get_value_ref_W<ViewYZ>().contrast.max = percent_min_max_[1];
        break;
    case WindowKind::ViewFilter2D:
        GSH::instance().get_view_cache().get_value_ref_W<ViewFilter2D>().contrast.min = percent_min_max_[0];
        GSH::instance().get_view_cache().get_value_ref_W<ViewFilter2D>().contrast.max = percent_min_max_[1];
        break;
    }
}
} // namespace holovibes::compute
