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
    if (view_cache_.get_value<Filter2D>().log_enabled)
        insert_filter2d_view_log();
}

void Rendering::insert_contrast()
{
    LOG_FUNC(compute_worker);

    // Compute min and max pixel values if requested
    insert_compute_autocontrast();

    // Apply contrast on the main view
    if (view_cache_.get_value<ViewXY>().contrast.enabled)
        insert_apply_contrast(WindowKind::XYview);

    // Apply contrast on cuts if needed
    if (view_cache_.get_value<CutsViewEnabled>())
    {
        if (view_cache_.get_value<ViewXZ>().contrast.enabled)
            insert_apply_contrast(WindowKind::XZview);
        if (view_cache_.get_value<ViewYZ>().contrast.enabled)

            insert_apply_contrast(WindowKind::YZview);
    }

    if (GSH::instance().get_value<Filter2DViewEnabled>() && GSH::instance().get_value<Filter2D>().contrast.enabled)
        insert_apply_contrast(WindowKind::Filter2D);
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
            case WindowKind::XYview:
                input = buffers_.gpu_postprocess_frame;
                size = buffers_.gpu_postprocess_frame_size;
                wind = view_cache_.get_value<ViewXY>();
                break;
            case WindowKind::YZview:
                input = buffers_.gpu_postprocess_frame_yz.get();
                size = fd_.height * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewYZ>();
                break;
            case WindowKind::XZview:
                input = buffers_.gpu_postprocess_frame_xz.get();
                size = fd_.width * compute_cache_.get_value<TimeTransformationSize>();
                wind = view_cache_.get_value<ViewXZ>();
                break;
            case WindowKind::Filter2D:
                input = buffers_.gpu_float_filter2d_frame.get();
                size = fd_.width * fd_.height;
                wind = GSH::instance().get_value<Filter2D>();
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

void Rendering::insert_compute_autocontrast()
{
    LOG_FUNC(compute_worker);

    // requested check are inside the lambda so that we don't need to
    // refresh the pipe at each autocontrast
    auto lambda_autocontrast = [&]()
    {
        // Compute autocontrast once the gpu time transformation queue is full
        if (!time_transformation_env_.gpu_time_transformation_queue->is_full())
            return;

        if (view_cache_.get_value<ViewXY>().get_exec_auto_contrast() &&
            (!image_acc_env_.gpu_accumulation_xy_queue || image_acc_env_.gpu_accumulation_xy_queue->is_full()))
        {
            // FIXME Handle composite size, adapt width and height (frames_res =
            // buffers_.gpu_postprocess_frame_size)
            autocontrast_caller(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, 0, WindowKind::XYview);
            view_cache_.get_value<ViewXY>().reset_exec_auto_contrast();
        }
        if (view_cache_.get_value<ViewXZ>().get_exec_auto_contrast() &&
            (!image_acc_env_.gpu_accumulation_xz_queue || image_acc_env_.gpu_accumulation_xz_queue->is_full()))
        {
            autocontrast_caller(buffers_.gpu_postprocess_frame_xz.get(),
                                fd_.width,
                                compute_cache_.get_value<TimeTransformationSize>(),
                                advanced_cache_.get_value<CutsContrastPOffset>(),
                                WindowKind::XZview);
            view_cache_.get_value<ViewXZ>().reset_exec_auto_contrast();
        }
        if (view_cache_.get_value<ViewYZ>().get_exec_auto_contrast() &&
            (!image_acc_env_.gpu_accumulation_yz_queue || image_acc_env_.gpu_accumulation_yz_queue->is_full()))
        {
            autocontrast_caller(buffers_.gpu_postprocess_frame_yz.get(),
                                compute_cache_.get_value<TimeTransformationSize>(),
                                fd_.height,
                                advanced_cache_.get_value<CutsContrastPOffset>(),
                                WindowKind::YZview);
            view_cache_.get_value<ViewYZ>().reset_exec_auto_contrast();
        }
        if (view_cache_.get_value<Filter2D>().get_exec_auto_contrast())
        {
            autocontrast_caller(buffers_.gpu_float_filter2d_frame.get(),
                                fd_.width,
                                fd_.height,
                                0,
                                WindowKind::Filter2D);
            view_cache_.get_value<Filter2D>().reset_exec_auto_contrast();
        }

        // FIXME: gsh should not be modified in the pipe
        view_cache_.synchronize<ViewPipeRequestOnSync>(api::get_compute_pipe());
    };

    fn_compute_vect_.conditional_push_back(lambda_autocontrast);
}

void Rendering::autocontrast_caller(
    float* input, const uint width, const uint height, const uint offset, WindowKind view)
{
    LOG_FUNC(compute_worker);

    constexpr uint percent_size = 2;

    const float percent_in[percent_size] = {advanced_cache_.get_value<ContrastLowerThreshold>(),
                                            advanced_cache_.get_value<ContrastUpperThreshold>()};
    switch (view)
    {
    case WindowKind::XYview:
    case WindowKind::XZview:
    case WindowKind::Filter2D:
        // No offset
        compute_percentile_xy_view(input,
                                   width,
                                   height,
                                   (view == WindowKind::XYview) ? 0 : offset,
                                   percent_in,
                                   percent_min_max_,
                                   percent_size,
                                   zone_cache_.get_value<ReticleZone>(),
                                   (view == WindowKind::Filter2D) ? false
                                                                  : view_cache_.get_value<ReticleDisplayEnabled>(),
                                   stream_);
        api::change_window(view)->set_contrast_min(percent_min_max_[0]).set_contrast_max(percent_min_max_[1]);
        break;
    case WindowKind::YZview: // TODO: finished refactoring to remove this switch
        compute_percentile_yz_view(input,
                                   width,
                                   height,
                                   offset,
                                   percent_in,
                                   percent_min_max_,
                                   percent_size,
                                   zone_cache_.get_value<ReticleZone>(),
                                   view_cache_.get_value<ReticleDisplayEnabled>(),
                                   stream_);
        api::change_window(view)->set_contrast_min(percent_min_max_[0]).set_contrast_max(percent_min_max_[1]);
        break;
    }
}
} // namespace holovibes::compute
