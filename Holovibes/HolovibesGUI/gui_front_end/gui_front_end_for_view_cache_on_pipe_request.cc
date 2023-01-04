#include "gui_front_end_for_view_cache_on_pipe_request.hh"
#include "pipe_request_on_sync.hh"
#include "user_interface.hh"
#include "API.hh"

namespace holovibes::gui
{

static bool guard_edit_view()
{
    if (!UserInterface::instance().get_update_window())
    {
        LOG_WARN("Stop change on window to avoid dead lock, retry change");
        return false;
    }
    return true;
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<ChartDisplayEnabled>()
{
    LOG_UPDATE_FRONT_END_BEFORE(ChartDisplayEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<ChartDisplayEnabled>())
        return;

    UserInterface::instance().main_window->synchronize_thread(
        []()
        {
            UserInterface::instance().plot_window_.reset(nullptr);
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<ChartDisplayEnabled>()
{
    LOG_UPDATE_FRONT_END_AFTER(ChartDisplayEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<ChartDisplayEnabled>() == false)
        return;

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().plot_window_ =
                std::make_unique<gui::PlotWindow>(*api::get_compute_pipe().get_chart_env().chart_display_queue_,
                                                  UserInterface::instance().auto_scale_point_threshold_,
                                                  "Chart");
            UserInterface::instance().can_edit_window();
        },
        true);

    // UserInterface::instance().get_export_panel()->connect(UserInterface::instance().plot_window_.get(),
    //                                                       SIGNAL(closed()),
    //                                                       UserInterface::instance().get_export_panel(),
    //                                                       SLOT(stop_chart_display()),
    //                                                       Qt::UniqueConnection);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<CutsViewEnabled>()
{
    LOG_UPDATE_FRONT_END_BEFORE(CutsViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<CutsViewEnabled>())
        return;

    UserInterface::instance().main_window->synchronize_thread(
        []()
        {
            UserInterface::instance().sliceXZ.reset(nullptr);
            UserInterface::instance().sliceYZ.reset(nullptr);
            if (UserInterface::instance().xy_window)
            {
                UserInterface::instance().xy_window->setCursor(Qt::ArrowCursor);
                UserInterface::instance().xy_window->getOverlayManager().disable_all(gui::KindOfOverlay::SliceCross);
                UserInterface::instance().xy_window->getOverlayManager().disable_all(gui::KindOfOverlay::Cross);
            }
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<CutsViewEnabled>()
{
    LOG_UPDATE_FRONT_END_AFTER(CutsViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<CutsViewEnabled>() == false)
        return;

    QPoint xzPos = UserInterface::instance().xy_window->framePosition() +
                   QPoint(0, UserInterface::instance().xy_window->height() + 42);
    QPoint yzPos = UserInterface::instance().xy_window->framePosition() +
                   QPoint(UserInterface::instance().xy_window->width() + 20, 0);

    // FIXME WHY -- FIXME API
    const ushort nImg = api::get_time_transformation_size();
    uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

    if (time_transformation_size > UserInterface::instance().get_view_panel()->time_transformation_cuts_window_max_size)
        time_transformation_size = UserInterface::instance().get_view_panel()->time_transformation_cuts_window_max_size;

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().sliceXZ.reset(new gui::SliceWindow(
                "XZ view",
                xzPos,
                QSize(UserInterface::instance().xy_window->width(), api::get_time_transformation_size()),
                api::get_compute_pipe().get_stft_slice_queue(0).get(),
                gui::KindOfView::SliceXZ));

            UserInterface::instance().sliceYZ.reset(new gui::SliceWindow(
                "YZ view",
                yzPos,
                QSize(api::get_time_transformation_size(), UserInterface::instance().xy_window->height()),
                api::get_compute_pipe().get_stft_slice_queue(1).get(),
                gui::KindOfView::SliceYZ));
            UserInterface::instance().xy_window->getOverlayManager().create_overlay<gui::KindOfOverlay::Cross>();

            auto holo = dynamic_cast<gui::HoloWindow*>(UserInterface::instance().xy_window.get());
            if (holo)
                holo->update_slice_transforms();
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<Reticle>()
{
    LOG_UPDATE_FRONT_END_BEFORE(Reticle);

    if (!UserInterface::instance().xy_window)
        return;

    if (api::get_reticle().display_enabled)
        return;

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().xy_window->getOverlayManager().disable_all(gui::KindOfOverlay::Reticle);
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<Reticle>()
{
    LOG_UPDATE_FRONT_END_AFTER(Reticle);
    if (!UserInterface::instance().xy_window)
        return;

    if (api::get_reticle().display_enabled == false)
        return;

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().xy_window->getOverlayManager().create_overlay<gui::KindOfOverlay::Reticle>();
            UserInterface::instance().xy_window->getOverlayManager().create_default();
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<LensViewEnabled>()
{
    LOG_UPDATE_FRONT_END_BEFORE(LensViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<LensViewEnabled>())
        return;

    UserInterface::instance().main_window->synchronize_thread(
        []()
        {
            UserInterface::instance().lens_window.reset(nullptr);
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<LensViewEnabled>()
{
    LOG_UPDATE_FRONT_END_AFTER(LensViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<LensViewEnabled>() == false)
        return;

    // Set positions of new windows according to the position of the main GL window
    QPoint pos = UserInterface::instance().xy_window->framePosition() +
                 QPoint(UserInterface::instance().xy_window->width() + 310, 0);

    const FrameDescriptor& fd = api::get_import_frame_descriptor();
    ushort lens_window_width = fd.width;
    ushort lens_window_height = fd.height;
    get_good_size(lens_window_width, lens_window_height, UserInterface::auxiliary_window_max_size);

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().lens_window.reset(
                new gui::RawWindow("Lens view",
                                   pos,
                                   QSize(lens_window_width, lens_window_height),
                                   api::get_compute_pipe().get_fourier_transforms().get_lens_queue().get(),
                                   0.f,
                                   gui::KindOfView::Lens));
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<RawViewEnabled>()
{
    LOG_UPDATE_FRONT_END_BEFORE(RawViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<RawViewEnabled>())
        return;

    UserInterface::instance().main_window->synchronize_thread(
        []()
        {
            UserInterface::instance().raw_window.reset(nullptr);
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<RawViewEnabled>()
{
    LOG_UPDATE_FRONT_END_AFTER(RawViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::detail::get_value<RawViewEnabled>() == false)
        return;

    const FrameDescriptor& fd = api::get_import_frame_descriptor();
    ushort raw_window_width = fd.width;
    ushort raw_window_height = fd.height;
    get_good_size(raw_window_width, raw_window_height, UserInterface::auxiliary_window_max_size);

    // Set positions of new windows according to the position of the main GL window and Raw window
    QPoint pos = UserInterface::instance().xy_window->framePosition() +
                 QPoint(UserInterface::instance().xy_window->width() + 310, 0);

    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().raw_window.reset(
                new gui::RawWindow("Raw view",
                                   pos,
                                   QSize(raw_window_width, raw_window_height),
                                   api::get_compute_pipe().get_raw_view_queue_ptr().get()));
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::before_method<Filter2DViewEnabled>()
{
    LOG_UPDATE_FRONT_END_BEFORE(Filter2DViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::get_filter2d_view_enabled())
        return;

    UserInterface::instance().main_window->synchronize_thread(
        []()
        {
            UserInterface::instance().filter2d_window.reset(nullptr);
            UserInterface::instance().can_edit_window();
        },
        true);
}

template <>
void GuiFrontEndForViewCacheOnPipeRequest::after_method<Filter2DViewEnabled>()
{
    LOG_UPDATE_FRONT_END_AFTER(Filter2DViewEnabled);
    if (guard_edit_view() == false)
    {
        return;
    }

    if (api::get_filter2d_view_enabled() == false)
        return;

    const FrameDescriptor& fd = api::get_import_frame_descriptor();
    ushort filter2d_window_width = fd.width;
    ushort filter2d_window_height = fd.height;
    get_good_size(filter2d_window_width, filter2d_window_height, UserInterface::instance().auxiliary_window_max_size);

    // set positions of new windows according to the position of the
    // main GL window
    QPoint pos = UserInterface::instance().xy_window->framePosition() +
                 QPoint(UserInterface::instance().xy_window->width() + 310, 0);
    UserInterface::instance().main_window->synchronize_thread(
        [=]()
        {
            UserInterface::instance().filter2d_window.reset(
                new Filter2DWindow("Filter2D view",
                                   pos,
                                   QSize(filter2d_window_width, filter2d_window_height),
                                   api::get_compute_pipe().get_filter2d_view_queue_ptr().get()));
            UserInterface::instance().can_edit_window();
        },
        true);
}

} // namespace holovibes::gui
