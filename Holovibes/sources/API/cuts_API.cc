#include "API.hh"

namespace holovibes::api
{

bool set_3d_cuts_view(uint time_transformation_size)
{
    api::detail::set_value<TimeTransformationCutsEnable>(true);
    while (api::get_compute_pipe().get_composite_cache().has_change_requested())
        continue;

    // FIXME API : Need to move this outside this (and this function must be useless)
    try
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(0, UserInterfaceDescriptor::instance().mainDisplay->height() + 42);
        QPoint yzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 20, 0);

        UserInterfaceDescriptor::instance().sliceXZ.reset(new gui::SliceWindow(
            xzPos,
            QSize(UserInterfaceDescriptor::instance().mainDisplay->width(), time_transformation_size),
            api::get_compute_pipe().get_stft_slice_queue(0).get(),
            gui::KindOfView::SliceXZ));
        UserInterfaceDescriptor::instance().sliceXZ->setTitle("XZ view");
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(api::get_view_xz().get_rotation());
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(api::get_view_xz().get_flip_enabled());

        UserInterfaceDescriptor::instance().sliceYZ.reset(new gui::SliceWindow(
            yzPos,
            QSize(time_transformation_size, UserInterfaceDescriptor::instance().mainDisplay->height()),
            api::get_compute_pipe().get_stft_slice_queue(1).get(),
            gui::KindOfView::SliceYZ));
        UserInterfaceDescriptor::instance().sliceYZ->setTitle("YZ view");
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(api::get_view_yz().get_rotation());
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(api::get_view_yz().get_flip_enabled());

        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Cross>();
        api::set_cuts_view_enabled(true);
        auto holo = dynamic_cast<gui::HoloWindow*>(UserInterfaceDescriptor::instance().mainDisplay.get());
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
    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        UserInterfaceDescriptor::instance().mainDisplay->setCursor(Qt::ArrowCursor);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Cross);
    }
}

} // namespace holovibes::api
