#include "API.hh"

namespace holovibes::api
{

inline void set_batch_size(int value)
{
    api::detail::set_value<BatchSize>(value);

    // FIXME : need all vars on MicroCache
    if (value > get_input_buffer_size())
        api::detail::set_value<BatchSize>(value);

    if (get_time_stride() < value)
        set_time_stride(value);
    // Go to lower multiple
    if (get_time_stride() % value != 0)
        set_time_stride(get_time_stride() - get_time_stride() % value);
}

inline void set_time_stride(int value)
{
    // FIXME: temporary fix due to ttstride change in pipe.make_request
    // std::lock_guard<std::mutex> lock(mutex_);
    api::detail::set_value<TimeStride>(value);

    if (get_batch_size() > value)
        return set_time_stride(get_batch_size());

    // Go to lower multiple
    if (value % get_batch_size() != 0)
        return set_time_stride(value - value % get_batch_size());
}

void cancel_time_transformation_cuts(std::function<void()> callback)
{
    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        UserInterfaceDescriptor::instance().mainDisplay->setCursor(Qt::ArrowCursor);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Cross);
    }

    api::get_compute_pipe().insert_fn_end_vect(callback);

    // Refresh pipe to remove cuts linked lambda from pipe
    pipe_refresh();
    api::set_cuts_view_enabled(false);
}

void set_raw_mode(uint window_max_size)
{
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);

    set_compute_mode(Computation::Raw);
    create_pipe(); // To remove ?

    UserInterfaceDescriptor::instance().mainDisplay.reset(
        new holovibes::gui::RawWindow(pos,
                                      size,
                                      get_gpu_input_queue_ptr().get(),
                                      static_cast<float>(width) / static_cast<float>(height)));
    UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
    std::string fd_info =
        std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
}

void update_batch_size(std::function<void()> notify_callback, const uint batch_size)
{
    if (batch_size == api::get_batch_size())
        return;

    api::set_batch_size(batch_size);

    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe_ptr().get()))
    {
        get_compute_pipe().insert_fn_end_vect(notify_callback);
    }
    else
    {
        LOG_INFO(main, "could not get pipe");
    }
}

void update_time_stride(std::function<void()> callback, const uint time_stride)
{
    api::get_compute_pipe().insert_fn_end_vect(callback);
}

void set_time_transformation_size(std::function<void()> callback)
{
    api::get_compute_pipe().insert_fn_end_vect(callback);
}

void toggle_renormalize(bool value)
{
    set_renorm_enabled(value);

    if (UserInterfaceDescriptor::instance().import_type_ != ImportType::None)
        get_compute_pipe().request_clear_img_acc();

    pipe_refresh();
}

} // namespace holovibes::api
