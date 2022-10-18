#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void start_chart_display()
{
    auto pipe = api::get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    UserInterfaceDescriptor::instance().plot_window_ =
        std::make_unique<gui::PlotWindow>(*api::get_compute_pipe()->get_chart_display_queue(),
                                          UserInterfaceDescriptor::instance().auto_scale_point_threshold_,
                                          "Chart");
}

void stop_chart_display()
{
    try
    {
        auto pipe = api::get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
    }

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
}

double get_rotation()
{
    if (!api::is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.rot;
}

bool get_flip_enabled()
{

    if (!api::is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.flip_enabled;
}

unsigned GSH::get_img_accu_level() const
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    auto w = reinterpret_cast<const View_XYZ&>(get_current_window());
    return w.img_accu_level;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display()
{
    return UserInterfaceDescriptor::instance().mainDisplay;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window()
{
    return UserInterfaceDescriptor::instance().raw_window;
}

} // namespace holovibes::api
