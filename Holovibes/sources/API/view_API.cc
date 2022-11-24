#include "API.hh"

namespace holovibes::api
{

void create_holo_window(ushort window_size)
{
    QPoint pos(0, 0);
    const FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_size);
    QSize size(width, height);
    init_image_mode(pos, size);

    try
    {
        UserInterface::instance().main_display.reset(
            new gui::HoloWindow(pos,
                                size,
                                get_gpu_output_queue_ptr().get(),
                                UserInterface::instance().sliceXZ,
                                UserInterface::instance().sliceYZ,
                                static_cast<float>(width) / static_cast<float>(height)));
        UserInterface::instance().main_display->set_is_resize(false);
        UserInterface::instance().main_display->setTitle(QString("XY view"));
        UserInterface::instance().main_display->resetTransform();
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
    if (UserInterface::instance().main_display)
    {
        old_scale = UserInterface::instance().main_display->getScale();
        old_translation = UserInterface::instance().main_display->getTranslate();
    }

    api::close_windows();
    api::close_critical_compute();

    set_image_type(static_cast<ImageTypeEnum>(index));

    try
    {
        api::create_pipe();
        api::create_holo_window(window_size);
        UserInterface::instance().main_display->setScale(old_scale);
        UserInterface::instance().main_display->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        UserInterface::instance().main_display.reset(nullptr);
        LOG_ERROR(main, "refresh_view_mode: {}", e.what());
    }
}

} // namespace holovibes::api
