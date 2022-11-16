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
                     const camera::FrameDescriptor& input_fd,
                     const camera::FrameDescriptor& output_fd,
                     const cudaStream_t& stream,
                     AdvancedCache::Cache& advanced_cache,
                     ComputeCache::Cache& compute_cache,
                     ExportCache::Cache& export_cache,
                     ViewCache::Cache& view_cache,
                     ZoneCache::Cache& zone_cache)
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
//    , cache_(cache)
{
    // Hold 2 float values (min and max)
    cudaXMallocHost(&percent_min_max_, 2 * sizeof(float));
}

Rendering::~Rendering() { cudaXFreeHost(percent_min_max_); }

void Rendering::insert_fft_shift()
{
    LOG_FUNC(compute_worker);

    if (view_cache_.get_value<FftShiftEnabled>())
    {
        if (view_cache_.get_value<ImageType>() == ImageTypeEnum::Composite)
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
    LOG_FUNC(compute_worker);

    if (view_cache_.get_value<ChartDisplayEnabled>() || export_cache_.get_value<ChartRecord>().is_enable())
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
                if (export_cache_.get_value<ChartRecord>().is_enable() &&
                    chart_env_.current_nb_point_to_record_left > 0)
                {
                    chart_env_.chart_record_queue_->push_back(point);
                    --chart_env_.current_nb_point_to_record_left;
                }
            });
    }
}

void Rendering::insert_log()
{
    LOG_FUNC(compute_worker);

    if (view_cache_.get_value<ViewXY>().log_enabled)
        insert_main_log();
    if (view_cache_.get_value<CutsViewEnabled>())
        insert_slice_log();
    if (view_cache_.get_value<ViewFilter2D>().log_enabled)
        insert_filter2d_view_log();
}

void Rendering::insert_contrast()
{
    LOG_FUNC(compute_worker);

    insert_auto_request_contrast();
    insert_request_exec_contrast();

    // Apply contrast on the main view
    if (view_cache_.get_value<ViewXY>().contrast.enabled)
        insert_apply_contrast(WindowKind::ViewXY);

    // Apply contrast on cuts if needed
    if (view_cache_.get_value<CutsViewEnabled>())
    {
        if (view_cache_.get_value<ViewXZ>().contrast.enabled)
            insert_apply_contrast(WindowKind::ViewXZ);
        if (view_cache_.get_value<ViewYZ>().contrast.enabled)

            insert_apply_contrast(WindowKind::ViewYZ);
    }

    if (GSH::instance().get_value<ViewFilter2D>().contrast.enabled && GSH::instance().get_value<Filter2DViewEnabled>())
        insert_apply_contrast(WindowKind::ViewFilter2D);
}

void Rendering::insert_main_log()
{
    LOG_FUNC(compute_worker);

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
    LOG_FUNC(compute_worker);

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
    LOG_FUNC(compute_worker);

    if (GSH::instance().get_value<Filter2DViewEnabled>())
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

void Rendering::insert_apply_contrast(WindowKind view)
{
    LOG_FUNC(compute_worker);

    fn_compute_vect_.conditional_push_back(
        [=]()
        {
            // Set parameters
            float* input = nullptr;
            uint size = 0;
            constexpr ushort dynamic_range = 65535;
            float min = 0;
            float max = 0;

            ViewWindow wind;
            switch (view)
            {
            case WindowKind::ViewXY:
                input = buffers_.gpu_postprocess_frame;
                size = buffers_.gpu_postprocess_frame_size;
                wind = view_cache_.get_value<ViewXY>();
                break;
            case WindowKind::ViewYZ:
                input = buffers_.gpu_postprocess_frame_yz.get();
                size = fd_.height * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewYZ>();
                break;
            case WindowKind::ViewXZ:
                input = buffers_.gpu_postprocess_frame_xz.get();
                size = fd_.width * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewXZ>();
                break;
            case WindowKind::ViewFilter2D:
                input = buffers_.gpu_float_filter2d_frame.get();
                size = fd_.width * fd_.height;
                wind = view_cache_.get_value<ViewFilter2D>();
                break;
            }

            if (wind.contrast.invert)
            {
                min = wind.contrast.max;
                max = wind.contrast.min;
            }
            else
            {
                min = wind.contrast.min;
                max = wind.contrast.max;
            }

            apply_contrast_correction(input, size, dynamic_range, min, max, stream_);
        });
}

void Rendering::insert_request_exec_contrast()
{
    LOG_FUNC(compute_worker);

    // requested check are inside the lambda so that we don't need to
    // refresh the pipe at each autocontrast
    auto lambda = [&]()
    {
        // Compute autocontrast once the gpu time transformation queue is full
        if (!time_transformation_env_.gpu_time_transformation_queue->is_full())
            return;

        if (has_requested_view_exec_contrast(WindowKind::ViewXY))
        {
            reset_view_exec_contrast(WindowKind::ViewXY);
            autocontrast_caller(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, 0, WindowKind::ViewXY);
        }

        if (api::get_cuts_view_enabled())
        {
            if (has_requested_view_exec_contrast(WindowKind::ViewXZ))
            {
                reset_view_exec_contrast(WindowKind::ViewXZ);
                autocontrast_caller(buffers_.gpu_postprocess_frame_xz.get(),
                                    fd_.width,
                                    compute_cache_.get_value<TimeTransformationSize>(),
                                    advanced_cache_.get_value<ContrastThreshold>().cuts_p_offset,
                                    WindowKind::ViewXZ);
            }
            if (has_requested_view_exec_contrast(WindowKind::ViewYZ))
            {
                reset_view_exec_contrast(WindowKind::ViewYZ);
                autocontrast_caller(buffers_.gpu_postprocess_frame_yz.get(),
                                    compute_cache_.get_value<TimeTransformationSize>(),
                                    fd_.height,
                                    advanced_cache_.get_value<ContrastThreshold>().cuts_p_offset,
                                    WindowKind::ViewYZ);
            }
        }

        if (api::get_filter2d_view_enabled() && has_requested_view_exec_contrast(WindowKind::ViewFilter2D))
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

void Rendering::insert_auto_request_contrast()
{
    LOG_FUNC(compute_worker);

    auto lambda = [&]()
    {
        if (api::get_view_xy().contrast.auto_refresh)
            request_view_exec_contrast(WindowKind::ViewXY);
        if (api::get_cuts_view_enabled())
        {
            if (api::get_view_xz().contrast.auto_refresh)
                request_view_exec_contrast(WindowKind::ViewXZ);
            if (api::get_view_yz().contrast.auto_refresh)
                request_view_exec_contrast(WindowKind::ViewYZ);
        }
        if (api::get_filter2d_view_enabled() && api::get_view_filter2d().contrast.auto_refresh)
            request_view_exec_contrast(WindowKind::ViewFilter2D);
    };

    fn_compute_vect_.conditional_push_back(lambda);
}

void Rendering::insert_clear_image_accumulation()
{
    LOG_FUNC(compute_worker);

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
    LOG_FUNC(compute_worker);

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
        api::change_window(view)->contrast.min = percent_min_max_[0];
        api::change_window(view)->contrast.max = percent_min_max_[1];
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
        api::change_window(view)->contrast.min = percent_min_max_[0];
        api::change_window(view)->contrast.max = percent_min_max_[1];
        break;
    }
}
} // namespace holovibes::compute
