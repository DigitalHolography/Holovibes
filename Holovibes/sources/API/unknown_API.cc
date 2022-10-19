#include "API.hh"

namespace holovibes::api
{

void check_q_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (api::get_view_accu_q().get_accu_level() > upper_bound)
        api::change_view_accu_q().set_accu_level(upper_bound);

    upper_bound -= api::get_view_accu_q().get_accu_level();

    if (upper_bound >= 0 && api::get_view_accu_q().get_index() > static_cast<uint>(upper_bound))
        api::change_view_accu_q().set_index(upper_bound);
}

void init_image_mode(QPoint& position, QSize& size)
{
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        position = UserInterfaceDescriptor::instance().mainDisplay->framePosition();
        size = UserInterfaceDescriptor::instance().mainDisplay->size();
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }
}

bool set_holographic_mode(ushort window_size)
{
    /* ---------- */
    try
    {
        set_compute_mode(Computation::Hologram);
        /* Pipe & Window */
        create_pipe();
        create_holo_window(window_size);
        /* Info Manager */
        auto fd = api::get_gpu_input_queue().get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        /* Contrast */
        GSH::instance().get_current_window().set_contrast_enabled(true);

        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR(main, "cannot set holographic mode: {}", e.what());
    }

    return false;
}

void close_critical_compute()
{
    if (get_convolution_enabled())
        api::disable_convolution();

    if (api::get_cuts_view_enabled())
        cancel_time_transformation_cuts([]() {});

    Holovibes::instance().stop_compute();
}

void open_advanced_settings(QMainWindow* parent, ::holovibes::gui::AdvancedSettingsWindowPanel* specific_panel)
{
    UserInterfaceDescriptor::instance().is_advanced_settings_displayed = true;
    UserInterfaceDescriptor::instance().advanced_settings_window_ =
        std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(parent, specific_panel);
}

} // namespace holovibes::api
