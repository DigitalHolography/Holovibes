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

    pipe_refresh();
}

void reticle_scale(float value)
{
    set_reticle_scale(value);
    pipe_refresh();
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
                                get_compute_pipe_ptr(),
                                UserInterfaceDescriptor::instance().sliceXZ,
                                UserInterfaceDescriptor::instance().sliceYZ,
                                static_cast<float>(width) / static_cast<float>(height)));
        UserInterfaceDescriptor::instance().mainDisplay->set_is_resize(false);
        UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
        UserInterfaceDescriptor::instance().mainDisplay->resetTransform();
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(api::get_rotation());
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(api::get_flip_enabled());
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

    set_img_type(static_cast<ImgType>(index));

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
    UserInterfaceDescriptor::instance().last_img_type_ = value;

    get_compute_pipe().insert_fn_end_vect(callback);
    pipe_refresh();

    // Force XYview autocontrast
    get_compute_pipe().request_autocontrast(WindowKind::XYview);
    // Force cuts views autocontrast if needed
}

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        get_compute_pipe().request_filter2d_view();
        while (get_compute_pipe().get_filter2d_view_requested())
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
                                    get_compute_pipe().get_filter2d_view_queue().get()));

        UserInterfaceDescriptor::instance().filter2d_window->setTitle("Filter2D view");

        GSH::instance().change_value<Filter2D>().set_log_scale_slice_enabled(true);
        get_compute_pipe().request_autocontrast(WindowKind::Filter2D);
    }
    else
    {
        UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);
        get_compute_pipe().request_disable_filter2d_view();
        while (get_compute_pipe().get_disable_filter2d_view_requested())
            continue;
    }

    pipe_refresh();
}

void set_filter2d(bool checked)
{
    set_filter2d_view_enabled(checked);
    set_auto_contrast_all();
}

void set_accumulation_level(int value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    reinterpret_cast<View_XYZ*>(api::get_current_window_ptr().get())->img_accu_level = value;
}

} // namespace holovibes::api
