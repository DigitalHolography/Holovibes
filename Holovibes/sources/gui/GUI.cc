#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include "API.hh"

#define UI UserInterfaceDescriptor::instance()

namespace holovibes::gui
{

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        const camera::FrameDescriptor& fd = api::get_fd();
        ushort filter2d_window_width = fd.width;
        ushort filter2d_window_height = fd.height;
        get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);
        UI.filter2d_window.reset(new gui::Filter2DWindow(pos,
                                                         QSize(filter2d_window_width, filter2d_window_height),
                                                         api::get_compute_pipe()->get_filter2d_view_queue().get()));

        UI.filter2d_window->setTitle("Filter2D view");
    }
    else
        UI.filter2d_window.reset(nullptr);
}

void set_lens_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 310, 0);

        const ::camera::FrameDescriptor& fd = api::get_fd();
        ushort lens_window_width = fd.width;
        ushort lens_window_height = fd.height;
        get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

        UI.lens_window.reset(new gui::RawWindow(pos,
                                                QSize(lens_window_width, lens_window_height),
                                                api::get_compute_pipe()->get_lens_queue().get(),
                                                0.f,
                                                gui::KindOfView::Lens));

        UI.lens_window->setTitle("Lens view");
    }
    else
        UI.lens_window.reset(nullptr);
}

void set_3d_cuts_view(bool checked, uint window_size)
{
    if (checked)
    {
        window_size = std::max(256u, std::min(512u, window_size));

        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UI.mainDisplay->framePosition() + QPoint(0, UI.mainDisplay->height() + 42);
        QPoint yzPos = UI.mainDisplay->framePosition() + QPoint(UI.mainDisplay->width() + 20, 0);

        LOG_ERROR("window_size: {}", window_size);

        UI.sliceXZ.reset(new gui::SliceWindow(xzPos,
                                              QSize(UI.mainDisplay->width(), window_size),
                                              api::get_compute_pipe()->get_stft_slice_queue(0).get(),
                                              gui::KindOfView::SliceXZ));
        UI.sliceXZ->setTitle("XZ view");
        UI.sliceXZ->setAngle(api::get_xz_rotation());
        UI.sliceXZ->setFlip(api::get_xz_horizontal_flip());

        UI.sliceYZ.reset(new gui::SliceWindow(yzPos,
                                              QSize(window_size, UI.mainDisplay->height()),
                                              api::get_compute_pipe()->get_stft_slice_queue(1).get(),
                                              gui::KindOfView::SliceYZ));
        UI.sliceYZ->setTitle("YZ view");
        UI.sliceYZ->setAngle(api::get_yz_rotation());
        UI.sliceYZ->setFlip(api::get_yz_horizontal_flip());

        UI.mainDisplay->getOverlayManager().create_overlay<gui::Cross>();

        api::set_cuts_view_enabled(true);
        api::set_yz_enabled(true);
        api::set_xz_enabled(true);

        api::set_auto_contrast_cuts();

        auto holo = dynamic_cast<gui::HoloWindow*>(UI.mainDisplay.get());
        if (holo)
            holo->update_slice_transforms();

        api::pipe_refresh();
    }
    else
    {
        UI.sliceXZ.reset(nullptr);
        UI.sliceYZ.reset(nullptr);

        if (UI.mainDisplay)
        {
            UI.mainDisplay->setCursor(Qt::ArrowCursor);
            UI.mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
            UI.mainDisplay->getOverlayManager().disable_all(gui::Cross);
        }
    }

    LOG_ERROR("3D cuts view enabled: {}", checked);
}

void set_composite_area() { UI.mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>(); }

void open_advanced_settings(QMainWindow* parent)
{
    UI.is_advanced_settings_displayed = true;
    UI.advanced_settings_window_ = std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(parent);
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display() { return UI.mainDisplay; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz() { return UI.sliceXZ; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz() { return UI.sliceYZ; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window() { return UI.lens_window; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window() { return UI.raw_window; }

std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window() { return UI.filter2d_window; }

} // namespace holovibes::gui