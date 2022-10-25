#include "API.hh"

namespace holovibes::api
{

const View_Window& get_window(WindowKind kind)
{
    if (kind == WindowKind::XYview)
        return static_cast<const View_Window&>(GSH::instance().get_view_cache().get_value<ViewXY>());
    else if (kind == WindowKind::XZview)
        return static_cast<const View_Window&>(GSH::instance().get_view_cache().get_value<ViewXZ>());
    else if (kind == WindowKind::YZview)
        return static_cast<const View_Window&>(GSH::instance().get_view_cache().get_value<ViewYZ>());
    else if (kind == WindowKind::Filter2D)
        return static_cast<const View_Window&>(GSH::instance().get_view_cache().get_value<Filter2D>());

    throw std::runtime_error("Unexpected WindowKind");
    // default case
    return static_cast<const View_Window&>(GSH::instance().get_view_cache().get_value<ViewXY>());
}

TriggerChangeValue<View_Window> change_window(WindowKind kind)
{
    if (kind == WindowKind::XYview)
        return GSH::instance().get_view_cache().change_value<ViewXY>();
    else if (kind == WindowKind::XZview)
        return GSH::instance().get_view_cache().change_value<ViewXZ>();
    else if (kind == WindowKind::YZview)
        return GSH::instance().get_view_cache().change_value<ViewYZ>();
    else if (kind == WindowKind::Filter2D)
        return GSH::instance().get_view_cache().change_value<Filter2D>();

    throw std::runtime_error("Unexpected WindowKind");
    return TriggerChangeValue<View_Window>([]() {}, nullptr);
}

void start_chart_display()
{
    api::get_compute_pipe().request_display_chart();
    // Wait for the chart display to be enabled for notify
    while (api::get_compute_pipe().get_chart_display_requested())
        continue;

    UserInterfaceDescriptor::instance().plot_window_ =
        std::make_unique<gui::PlotWindow>(*api::get_compute_pipe().get_chart_display_queue(),
                                          UserInterfaceDescriptor::instance().auto_scale_point_threshold_,
                                          "Chart");
}

void stop_chart_display()
{
    try
    {
        api::get_compute_pipe().request_disable_display_chart();
        // Wait for the chart display to be disabled for notify
        while (api::get_compute_pipe().get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
    }

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display()
{
    return UserInterfaceDescriptor::instance().mainDisplay;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window()
{
    return UserInterfaceDescriptor::instance().raw_window;
}

void set_raw_view(bool checked, uint auxiliary_window_max_size)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    if (checked)
    {
        {
            GSH::instance().set_value<RawViewEnabled>(true);
        }
        while (ViewCache::RefSingleton::has_change())
            continue;

        const ::camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                     QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
        UserInterfaceDescriptor::instance().raw_window.reset(
            new gui::RawWindow(pos,
                               QSize(raw_window_width, raw_window_height),
                               get_compute_pipe().get_raw_view_queue().get()));

        UserInterfaceDescriptor::instance().raw_window->setTitle("Raw view");
    }
    else
    {
        UserInterfaceDescriptor::instance().raw_window.reset(nullptr);

        {
            GSH::instance().set_value<RawViewEnabled>(false);
        }
        while (ViewCache::RefSingleton::has_change())
            continue;
    }

    pipe_refresh();
}

void set_lens_view(bool checked, uint auxiliary_window_max_size)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    set_lens_view_enabled(checked);

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                         QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);

            const ::camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;

            get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

            UserInterfaceDescriptor::instance().lens_window.reset(
                new gui::RawWindow(pos,
                                   QSize(lens_window_width, lens_window_height),
                                   get_compute_pipe().get_lens_queue().get(),
                                   0.f,
                                   gui::KindOfView::Lens));

            UserInterfaceDescriptor::instance().lens_window->setTitle("Lens view");
        }
        catch (const std::exception& e)
        {
            LOG_ERROR(main, "Catch {}", e.what());
        }
    }
    else
    {
        UserInterfaceDescriptor::instance().lens_window.reset(nullptr);

        get_compute_pipe().request_disable_lens_view();
        while (get_compute_pipe().get_disable_lens_view_requested())
            continue;

        pipe_refresh();
    }
}

void close_windows()
{
    UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);

    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);
    UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);

    if (UserInterfaceDescriptor::instance().lens_window)
        set_lens_view(false, 0);
    if (UserInterfaceDescriptor::instance().raw_window)
        set_raw_view(false, 0);

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
}

} // namespace holovibes::api
