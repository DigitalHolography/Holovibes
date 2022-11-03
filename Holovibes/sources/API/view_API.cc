#include "API.hh"

namespace holovibes::api
{
void display_reticle(bool value)
{
    set_reticle_display_enabled(value);

    if (value)
    {
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Reticle>();
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_default();
    }
    else
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Reticle);
}

void create_holo_window(ushort window_size)
{
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_size);
    QSize size(width, height);
    init_image_mode(pos, size);

    try
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(
            new gui::HoloWindow(pos,
                                size,
                                get_gpu_output_queue_ptr().get(),
                                UserInterfaceDescriptor::instance().sliceXZ,
                                UserInterfaceDescriptor::instance().sliceYZ,
                                static_cast<float>(width) / static_cast<float>(height)));
        UserInterfaceDescriptor::instance().mainDisplay->set_is_resize(false);
        UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
        UserInterfaceDescriptor::instance().mainDisplay->resetTransform();
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(api::get_current_window_as_view_xyz().get_rotation());
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(
            api::get_current_window_as_view_xyz().get_flip_enabled());
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR(main, "create_holo_window: {}", e.what());
    }
}

void refresh_view_mode(ushort window_size, uint index)
{
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        old_scale = UserInterfaceDescriptor::instance().mainDisplay->getScale();
        old_translation = UserInterfaceDescriptor::instance().mainDisplay->getTranslate();
    }

    api::close_windows();
    api::close_critical_compute();

    set_image_type(static_cast<ImageTypeEnum>(index));

    try
    {
        api::create_pipe();
        api::create_holo_window(window_size);
        UserInterfaceDescriptor::instance().mainDisplay->setScale(old_scale);
        UserInterfaceDescriptor::instance().mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        LOG_ERROR(main, "refresh_view_mode: {}", e.what());
    }
}

void set_view_mode(const std::string& value, std::function<void()> callback)
{
    api::detail::set_value<LastImageType>(value);
    get_compute_pipe().insert_fn_end_vect(callback);
    api::get_view_xy().request_exec_auto_contrast();
}

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        api::detail::set_value<Filter2DViewEnabled>(true);
        while (api::get_compute_pipe().get_view_cache().has_change_requested())
            continue;

        const camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
        ushort filter2d_window_width = fd.width;
        ushort filter2d_window_height = fd.height;
        get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                     QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
        UserInterfaceDescriptor::instance().filter2d_window.reset(
            new gui::Filter2DWindow(pos,
                                    QSize(filter2d_window_width, filter2d_window_height),
                                    get_compute_pipe().get_filter2d_view_queue_ptr().get()));

        UserInterfaceDescriptor::instance().filter2d_window->setTitle("Filter2D_PARAM view");

        GSH::instance().change_value<Filter2D_PARAM>()->set_log_scale_slice_enabled(true);
        api::get_view_filter2d().request_exec_auto_contrast();
    }
    else
    {
        UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);

        api::detail::set_value<Filter2DViewEnabled>(false);
        while (api::get_compute_pipe().get_view_cache().has_change_requested())
            continue;
    }
}

void set_filter2d(bool checked)
{
    set_filter2d_view_enabled(checked);
    set_auto_contrast_all();
}

} // namespace holovibes::api
