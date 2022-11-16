#include "API.hh"

namespace holovibes
{

static void allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
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

// FIXME API : these 3 function need to use the same function
static void view_xy_refresh(const ViewXYZ& new_value, Pipe& pipe)
{
    if (new_value.is_image_accumulation_enabled() == false)
        pipe.get_image_acc_env().gpu_accumulation_xy_queue.reset(nullptr);
    else
    {
        auto new_fd = api::get_gpu_input_queue().get_fd();
        new_fd.depth =
            GSH::instance().get_value<ImageType>() == ImageTypeEnum::Composite ? 3 * sizeof(float) : sizeof(float);
        allocate_accumulation_queue(pipe.get_image_acc_env().gpu_accumulation_xy_queue,
                                    pipe.get_image_acc_env().gpu_float_average_xy_frame,
                                    GSH::instance().get_value<ViewXY>().img_accu_level,
                                    new_fd);
    }
}

template <>
void ViewPipeRequestOnSync::on_sync<ViewXY>(const ViewXYZ& new_value, const ViewXYZ& old_value, Pipe& pipe)
{
    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        view_xy_refresh(new_value, pipe);
    }
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<ViewXY>(const ViewXYZ& new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ViewXY);
    view_xy_refresh(new_value, pipe);
    request_pipe_refresh();
}

static void view_xz_refresh(const ViewXYZ& new_value, Pipe& pipe)
{
    if (new_value.is_image_accumulation_enabled() == false)
        pipe.get_image_acc_env().gpu_accumulation_xz_queue.reset(nullptr);
    else
    {
        auto new_fd = api::get_gpu_input_queue().get_fd();
        new_fd.depth = sizeof(float);
        new_fd.height = GSH::instance().get_value<TimeTransformationSize>();
        allocate_accumulation_queue(pipe.get_image_acc_env().gpu_accumulation_xz_queue,
                                    pipe.get_image_acc_env().gpu_float_average_xz_frame,
                                    GSH::instance().get_value<ViewXZ>().img_accu_level,
                                    new_fd);
    }
}

template <>
void ViewPipeRequestOnSync::on_sync<ViewXZ>(const ViewXYZ& new_value, const ViewXYZ& old_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ViewXZ);
    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        view_xz_refresh(new_value, pipe);
    }
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<ViewXZ>(const ViewXYZ& new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ViewXZ);
    view_xz_refresh(new_value, pipe);
    request_pipe_refresh();
}

static void view_yz_refresh(const ViewXYZ& new_value, Pipe& pipe)
{
    if (new_value.is_image_accumulation_enabled() == false)
        pipe.get_image_acc_env().gpu_accumulation_yz_queue.reset(nullptr);
    else
    {
        auto new_fd = api::get_gpu_input_queue().get_fd();
        new_fd.depth = sizeof(float);
        new_fd.width = GSH::instance().get_value<TimeTransformationSize>();
        allocate_accumulation_queue(pipe.get_image_acc_env().gpu_accumulation_yz_queue,
                                    pipe.get_image_acc_env().gpu_float_average_yz_frame,
                                    GSH::instance().get_value<ViewYZ>().img_accu_level,
                                    new_fd);
    }
}

template <>
void ViewPipeRequestOnSync::on_sync<ViewYZ>(const ViewXYZ& new_value, const ViewXYZ& old_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ViewXZ);
    if (new_value.is_image_accumulation_enabled() != old_value.is_image_accumulation_enabled())
    {
        view_yz_refresh(new_value, pipe);
    }
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<ViewYZ>(const ViewXYZ& new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ViewYZ);
    view_yz_refresh(new_value, pipe);
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<RawViewEnabled>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(RawViewEnabled);

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
void ViewPipeRequestOnSync::operator()<ChartDisplayEnabled>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(ChartDisplayEnabled);

    if (new_value == false)
        pipe.get_chart_env().chart_display_queue_.reset(nullptr);
    else
        pipe.get_chart_env().chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());

    // IDK but maybe ...
    request_pipe_refresh();
}

template <>
void ViewPipeRequestOnSync::operator()<Filter2DViewEnabled>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(Filter2DViewEnabled);

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
void ViewPipeRequestOnSync::operator()<LensViewEnabled>(bool new_value, Pipe& pipe)
{
    LOG_UPDATE_PIPE(LensViewEnabled);

    if (new_value == false)
        pipe.get_fourier_transforms().get_lens_queue().reset(nullptr);

    request_pipe_refresh();
}
} // namespace holovibes
