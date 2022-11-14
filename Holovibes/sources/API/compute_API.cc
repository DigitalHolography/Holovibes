#include "API.hh"

namespace holovibes::api
{

void set_batch_size(int value)
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

void set_time_stride(int value)
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

void toggle_renormalize(bool value)
{
    set_renorm_enabled(value);

    if (api::get_import_type() != ImportTypeEnum::None)
    {
        GSH::instance().get_view_cache().get_value_ref_W<ViewXY>().request_clear_image_accumulation();
        GSH::instance().get_view_cache().get_value_ref_W<ViewXZ>().request_clear_image_accumulation();
        GSH::instance().get_view_cache().get_value_ref_W<ViewYZ>().request_clear_image_accumulation();
    }
}

} // namespace holovibes::api
