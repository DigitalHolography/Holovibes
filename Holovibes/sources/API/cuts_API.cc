#include "API.hh"

namespace holovibes::api
{

bool set_3d_cuts_view(uint time_transformation_size)
{
    api::detail::set_value<TimeTransformationCutsEnable>(true);
    while (api::get_compute_pipe().get_compute_cache().has_change_requested())
        continue;

    // FIXME API : Need to move this outside this (and this function must be useless)
    try
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UserInterface::instance().main_display->framePosition() +
                       QPoint(0, UserInterface::instance().main_display->height() + 42);
        QPoint yzPos = UserInterface::instance().main_display->framePosition() +
                       QPoint(UserInterface::instance().main_display->width() + 20, 0);

        UserInterface::instance().sliceXZ.reset(
            new gui::SliceWindow(xzPos,
                                 QSize(UserInterface::instance().main_display->width(), time_transformation_size),
                                 api::get_compute_pipe().get_stft_slice_queue(0).get(),
                                 gui::KindOfView::SliceXZ));
        UserInterface::instance().sliceXZ->setTitle("XZ view");

        UserInterface::instance().sliceYZ.reset(
            new gui::SliceWindow(yzPos,
                                 QSize(time_transformation_size, UserInterface::instance().main_display->height()),
                                 api::get_compute_pipe().get_stft_slice_queue(1).get(),
                                 gui::KindOfView::SliceYZ));
        UserInterface::instance().sliceYZ->setTitle("YZ view");

        UserInterface::instance().main_display->getOverlayManager().create_overlay<gui::KindOfOverlay::Cross>();
        api::set_cuts_view_enabled(true);
        auto holo = dynamic_cast<gui::HoloWindow*>(UserInterface::instance().main_display.get());
        if (holo)
            holo->update_slice_transforms();
        return true;
    }
    catch (const std::logic_error& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
    }

    return false;
}

void cancel_time_transformation_cuts(std::function<void()> callback)
{
    api::get_compute_pipe().insert_fn_end_vect(callback);
    api::set_cuts_view_enabled(false);

    // FIXME API : Need to move this outside this (and this function must be useless)
    UserInterface::instance().sliceXZ.reset(nullptr);
    UserInterface::instance().sliceYZ.reset(nullptr);

    if (UserInterface::instance().main_display)
    {
        UserInterface::instance().main_display->setCursor(Qt::ArrowCursor);
        UserInterface::instance().main_display->getOverlayManager().disable_all(gui::KindOfOverlay::SliceCross);
        UserInterface::instance().main_display->getOverlayManager().disable_all(gui::KindOfOverlay::Cross);
    }
}

} // namespace holovibes::api
