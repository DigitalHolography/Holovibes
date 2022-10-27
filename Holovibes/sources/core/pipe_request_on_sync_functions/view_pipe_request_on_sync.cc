#include "API.hh"

namespace holovibes
{

// FIXME API : these 3 function need to use the same function
template <>
void ViewPipeRequestOnSync::operator()<ViewXY>(const View_XYZ& new_value, const View_XYZ& old_value, Pipe& pipe)
{
    if (new_value.get_request_clear_image_accumulation() == true)
    {
        if (new_value.is_image_accumulation_enabled())
            image_acc_env_.gpu_accumulation_xy_queue->clear();
        new_value.reset_request_clear_image_accumulation();
    }

    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        if (new_value.is_image_accumulation_enabled() == false)
            image_acc_env_.gpu_accumulation_xy_queue.reset(nullptr);
        else
        {
            auto new_fd = api::get_gpu_input_queue().get_fd();
            new_fd.depth =
                GSH::instance().get_value<ImgTypeParam>() == ImgType::Composite ? 3 * sizeof(float) : sizeof(float);
            allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xy_queue,
                                        image_acc_env_.gpu_float_average_xy_frame,
                                        GSH::instance().get_value<ViewXY>().get_img_accu_level(),
                                        new_fd);
        }
    }
}

template <>
void ViewPipeRequestOnSync::operator()<ViewXZ>(const View_XYZ&, const View_XYZ&, Pipe& pipe)
{
    if (new_value.get_request_clear_image_accumulation() == true)
    {
        if (new_value.is_image_accumulation_enabled())
            image_acc_env_.gpu_accumulation_xz_queue->clear();
        new_value.reset_request_clear_image_accumulation();
    }

    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        if (new_value.is_image_accumulation_enabled() == false)
            image_acc_env_.gpu_accumulation_xz_queue.reset(nullptr);
        else
        {
            auto new_fd = api::get_gpu_input_queue().get_fd();
            new_fd.depth = sizeof(float);
            new_fd.height = GSH::instance().get_value<TimeTransformationSize>();
            allocate_accumulation_queue(image_acc_env_.gpu_accumulation_xz_queue,
                                        image_acc_env_.gpu_float_average_xz_frame,
                                        GSH::instance().get_value<ViewXZ>().get_img_accu_level(),
                                        new_fd);
        }
    }
}

template <>
void ViewPipeRequestOnSync::operator()<ViewYZ>(const View_XYZ&, const View_XYZ&, Pipe& pipe)
{
    if (new_value.get_request_clear_image_accumulation() == true)
    {
        if (new_value.is_image_accumulation_enabled())
            image_acc_env_.gpu_accumulation_yz_queue->clear();
        new_value.reset_request_clear_image_accumulation();
    }

    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        if (new_value.is_image_accumulation_enabled() == false)
            image_acc_env_.gpu_accumulation_yz_queue.reset(nullptr);
        else
        {
            auto new_fd = api::get_gpu_input_queue().get_fd();
            new_fd.depth = sizeof(float);
            new_fd.width = GSH::instance().get_value<TimeTransformationSize>();
            allocate_accumulation_queue(image_acc_env_.gpu_accumulation_yz_queue,
                                        image_acc_env_.gpu_float_average_yz_frame,
                                        GSH::instance().get_value<ViewYZ>().get_img_accu_level(),
                                        new_fd);
        }
    }
}

template <>
void ViewPipeRequestOnSync::operator()<RawViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE RawViewEnabled");

    if (new_value == false)
        pipe.get_raw_view_queue_ptr().reset(nullptr);
    else
    {
        auto fd = pipe.get_gpu_input_queue().get_fd();
        pipe.get_raw_view_queue_ptr().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
    }

    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<ChartDisplayEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE ChartDisplayEnabled");

    if (new_value == false)
        pipe.get_chart_env().chart_display_queue_.reset(nullptr);
    else
        pipe.get_chart_env().chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());

    // IDK but maybe ...
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<Filter2DViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Filter2DViewEnabled");

    if (new_value == false)
        pipe.get_filter2d_view_queue_ptr().reset(nullptr);
    else
    {
        auto fd = pipe.get_gpu_output_queue().get_fd();
        pipe.get_filter2d_view_queue_ptr().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
    }

    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<LensViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE LensViewEnabled");

    if (new_value == false)
        pipe.get_fourier_transforms().get_lens_queue().reset(nullptr);

    request_pipe_refresh();
}
} // namespace holovibes
