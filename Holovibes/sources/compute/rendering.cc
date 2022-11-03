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
                     ComputeCache::Cache& compute_cache,
                     ExportCache::Cache& export_cache,
                     ViewCache::Cache& view_cache,
                     AdvancedCache::Cache& advanced_cache,
                     ZoneCache::Cache& zone_cache)
    : fn_compute_vect_(fn_compute_vect)
    , buffers_(buffers)
    , chart_env_(chart_env)
    , time_transformation_env_(time_transformation_env)
    , image_acc_env_(image_acc_env)
    , input_fd_(input_fd)
    , fd_(output_fd)
    , stream_(stream)
    , compute_cache_(compute_cache)
    , export_cache_(export_cache)
    , view_cache_(view_cache)
    , advanced_cache_(advanced_cache)
    , zone_cache_(zone_cache)
{
    // Hold 2 float values (min and max)
    cudaXMallocHost(&percent_min_max_, 2 * sizeof(float));
}

Rendering::~Rendering() { cudaXFreeHost(percent_min_max_); }

void Rendering::insert_fft_shift()
{
    LOG_FUNC();

    if (view_cache_.get_fft_shift_enabled())
    {
        if (view_cache_.get_img_type() == ImgType::Composite)
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

    if (view_cache_.get_chart_display_enabled() || export_cache_.get_chart_record_enabled())
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                auto signal_zone = zone_cache_.get_signal_zone();
                auto noise_zone = zone_cache_.get_noise_zone();

                if (signal_zone.width() == 0 || signal_zone.height() == 0 || noise_zone.width() == 0 ||
                    noise_zone.height() == 0)
                    return;

                ChartPoint point = make_chart_plot(buffers_.gpu_postprocess_frame,
                                                   input_fd_.width,
                                                   input_fd_.height,
                                                   signal_zone,
                                                   noise_zone,
                                                   stream_);

                if (view_cache_.get_chart_display_enabled())
                    chart_env_.chart_display_queue_->push_back(point);
                if (export_cache_.get_chart_record_enabled() && chart_env_.nb_chart_points_to_record_ != 0)
                {
                    chart_env_.chart_record_queue_->push_back(point);
                    --chart_env_.nb_chart_points_to_record_;
                }
            });
    }
}

void Rendering::insert_log()
{
    LOG_FUNC();

    if (view_cache_.get_xy().log_enabled)
        insert_main_log();
    if (view_cache_.get_cuts_view_enabled())
        insert_slice_log();
    if (view_cache_.get_filter2d().log_enabled)
        insert_filter2d_view_log();
}

void Rendering::insert_contrast(std::atomic<bool>& autocontrast_request,
                                std::atomic<bool>& autocontrast_slice_xz_request,
                                std::atomic<bool>& autocontrast_slice_yz_request,
                                std::atomic<bool>& autocontrast_filter2d_request)
{
    LOG_FUNC();

    // Compute min and max pixel values if requested
    insert_compute_autocontrast(autocontrast_request,
                                autocontrast_slice_xz_request,
                                autocontrast_slice_yz_request,
                                autocontrast_filter2d_request);

    // Apply contrast on the main view
    if (view_cache_.get_xy().contrast.enabled)
        insert_apply_contrast(WindowKind::XYview);

    // Apply contrast on cuts if needed
    if (view_cache_.get_cuts_view_enabled())
    {
        if (view_cache_.get_xz().contrast.enabled)
            insert_apply_contrast(WindowKind::XZview);
        if (view_cache_.get_yz().contrast.enabled)

            insert_apply_contrast(WindowKind::YZview);
    }

    if (GSH::instance().get_filter2d_view_enabled() && view_cache_.get_filter2d().contrast.enabled)
        insert_apply_contrast(WindowKind::Filter2D);
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

    if (view_cache_.get_xz().log_enabled)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                map_log10(buffers_.gpu_postprocess_frame_xz.get(),
                          buffers_.gpu_postprocess_frame_xz.get(),
                          fd_.width * compute_cache_.get_time_transformation_size(),
                          stream_);
            });
    }
    if (view_cache_.get_yz().log_enabled)
    {
        fn_compute_vect_.conditional_push_back(
            [=]()
            {
                map_log10(buffers_.gpu_postprocess_frame_yz.get(),
                          buffers_.gpu_postprocess_frame_yz.get(),
                          fd_.height * compute_cache_.get_time_transformation_size(),
                          stream_);
            });
    }
}

void Rendering::insert_filter2d_view_log()
{
    LOG_FUNC();

    if (GSH::instance().get_filter2d_view_enabled())
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
    LOG_FUNC();

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
                wind = view_cache_.get_xy();
                break;
            case WindowKind::YZview:
                input = buffers_.gpu_postprocess_frame_yz.get();
                size = fd_.height * compute_cache_.get_time_transformation_size();
                wind = view_cache_.get_yz();
                break;
            case WindowKind::XZview:
                input = buffers_.gpu_postprocess_frame_xz.get();
                size = fd_.width * compute_cache_.get_time_transformation_size();
                wind = view_cache_.get_xz();
                break;
            case WindowKind::Filter2D:
                input = buffers_.gpu_float_filter2d_frame.get();
                size = fd_.width * fd_.height;
                wind = view_cache_.get_filter2d();
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

void Rendering::insert_compute_autocontrast(std::atomic<bool>& autocontrast_request,
                                            std::atomic<bool>& autocontrast_slice_xz_request,
                                            std::atomic<bool>& autocontrast_slice_yz_request,
                                            std::atomic<bool>& autocontrast_filter2d_request)
{
    LOG_FUNC();

    // requested check are inside the lambda so that we don't need to
    // refresh the pipe at each autocontrast
    auto lambda_autocontrast = [&]()
    {
        // Compute autocontrast once the gpu time transformation queue is full
        if (!time_transformation_env_.gpu_time_transformation_queue->is_full())
            return;

        if (autocontrast_request &&
            (!image_acc_env_.gpu_accumulation_xy_queue || image_acc_env_.gpu_accumulation_xy_queue->is_full()))
        {
            // FIXME Handle composite size, adapt width and height (frames_res =
            // buffers_.gpu_postprocess_frame_size)
            autocontrast_caller(buffers_.gpu_postprocess_frame.get(), fd_.width, fd_.height, 0, WindowKind::XYview);
            autocontrast_request = false;
        }
        if (autocontrast_slice_xz_request &&
            (!image_acc_env_.gpu_accumulation_xz_queue || image_acc_env_.gpu_accumulation_xz_queue->is_full()))
        {
            autocontrast_caller(buffers_.gpu_postprocess_frame_xz.get(),
                                fd_.width,
                                compute_cache_.get_time_transformation_size(),
                                advanced_cache_.get_cuts_contrast_p_offset(),
                                WindowKind::XZview);
            autocontrast_slice_xz_request = false;
        }
        if (autocontrast_slice_yz_request &&
            (!image_acc_env_.gpu_accumulation_yz_queue || image_acc_env_.gpu_accumulation_yz_queue->is_full()))
        {
            autocontrast_caller(buffers_.gpu_postprocess_frame_yz.get(),
                                compute_cache_.get_time_transformation_size(),
                                fd_.height,
                                advanced_cache_.get_cuts_contrast_p_offset(),
                                WindowKind::YZview);
            autocontrast_slice_yz_request = false;
        }
        if (autocontrast_filter2d_request)
        {
            autocontrast_caller(buffers_.gpu_float_filter2d_frame.get(),
                                fd_.width,
                                fd_.height,
                                0,
                                WindowKind::Filter2D);
            autocontrast_filter2d_request = false;
        }

        view_cache_.synchronize(); // FIXME: gsh should not be modified in the pipe
    };

    fn_compute_vect_.conditional_push_back(lambda_autocontrast);
}

void Rendering::autocontrast_caller(
    float* input, const uint width, const uint height, const uint offset, WindowKind view)
{
    LOG_FUNC();

    constexpr uint percent_size = 2;

    const float percent_in[percent_size] = {advanced_cache_.get_contrast_lower_threshold(),
                                            advanced_cache_.get_contrast_upper_threshold()};
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
                                   zone_cache_.get_reticle_zone(),
                                   (view == WindowKind::Filter2D) ? false : view_cache_.get_reticle_display_enabled(),
                                   stream_);
        GSH::instance().update_contrast(view, percent_min_max_[0], percent_min_max_[1]);
        break;
    case WindowKind::YZview: // TODO: finished refactoring to remove this switch
        compute_percentile_yz_view(input,
                                   width,
                                   height,
                                   offset,
                                   percent_in,
                                   percent_min_max_,
                                   percent_size,
                                   zone_cache_.get_reticle_zone(),
                                   view_cache_.get_reticle_display_enabled(),
                                   stream_);
        GSH::instance().update_contrast(view, percent_min_max_[0], percent_min_max_[1]);
        break;
    }
}
} // namespace holovibes::compute
