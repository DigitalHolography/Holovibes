#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
    , parent_(find_main_window(parent))
{
}

ViewPanel::~ViewPanel() {}

void ViewPanel::set_view_mode(const QString& value) { parent_->set_view_image_type(value); }

void ViewPanel::set_unwrapping_2d(const bool value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->holovibes_.get_compute_pipe()->request_unwrapping_2d(value);
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::toggle_time_transformation_cuts(bool checked)
{
    QComboBox* winSelection = parent_->ui.WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());
    if (checked)
    {
        try
        {
            parent_->holovibes_.get_compute_pipe()->create_stft_slice_queue();
            // set positions of new windows according to the position of the
            // main GL window
            QPoint xzPos = parent_->mainDisplay->framePosition() + QPoint(0, parent_->mainDisplay->height() + 42);
            QPoint yzPos = parent_->mainDisplay->framePosition() + QPoint(parent_->mainDisplay->width() + 20, 0);
            const ushort nImg = parent_->cd_.time_transformation_size;
            uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

            if (time_transformation_size > parent_->time_transformation_cuts_window_max_size)
                time_transformation_size = parent_->time_transformation_cuts_window_max_size;

            while (parent_->holovibes_.get_compute_pipe()->get_update_time_transformation_size_request())
                continue;
            while (parent_->holovibes_.get_compute_pipe()->get_cuts_request())
                continue;
            parent_->sliceXZ.reset(
                new SliceWindow(xzPos,
                                QSize(parent_->mainDisplay->width(), time_transformation_size),
                                parent_->holovibes_.get_compute_pipe()->get_stft_slice_queue(0).get(),
                                KindOfView::SliceXZ,
                                parent_));
            parent_->sliceXZ->setTitle("XZ view");
            parent_->sliceXZ->setAngle(parent_->xzAngle);
            parent_->sliceXZ->setFlip(parent_->xzFlip);
            parent_->sliceXZ->setCd(&(parent_->cd_));

            parent_->sliceYZ.reset(
                new SliceWindow(yzPos,
                                QSize(time_transformation_size, parent_->mainDisplay->height()),
                                parent_->holovibes_.get_compute_pipe()->get_stft_slice_queue(1).get(),
                                KindOfView::SliceYZ,
                                parent_));
            parent_->sliceYZ->setTitle("YZ view");
            parent_->sliceYZ->setAngle(parent_->yzAngle);
            parent_->sliceYZ->setFlip(parent_->yzFlip);
            parent_->sliceYZ->setCd(&(parent_->cd_));

            parent_->mainDisplay->getOverlayManager().create_overlay<Cross>();
            parent_->cd_.set_time_transformation_cuts_enabled(true);
            set_auto_contrast_cuts();
            auto holo = dynamic_cast<HoloWindow*>(parent_->mainDisplay.get());
            if (holo)
                holo->update_slice_transforms();
            parent_->notify();
        }
        catch (const std::logic_error& e)
        {
            LOG_ERROR << e.what() << std::endl;
            cancel_stft_slice_view();
        }
    }
    else
    {
        cancel_stft_slice_view();
    }
}

void ViewPanel::cancel_stft_slice_view()
{
    parent_->cd_.reset_slice_view();
    parent_->sliceXZ.reset(nullptr);
    parent_->sliceYZ.reset(nullptr);

    if (parent_->mainDisplay)
    {
        parent_->mainDisplay->setCursor(Qt::ArrowCursor);
        parent_->mainDisplay->getOverlayManager().disable_all(SliceCross);
        parent_->mainDisplay->getOverlayManager().disable_all(Cross);
    }
    if (auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get()))
    {
        pipe->insert_fn_end_vect([=]() {
            parent_->cd_.set_time_transformation_cuts_enabled(false);
            pipe->delete_stft_slice_queue();

            parent_->ui.TimeTransformationCutsCheckBox->setChecked(false);
            parent_->notify();
        });
    }
}

void ViewPanel::cancel_time_transformation_cuts()
{
    if (parent_->cd_.time_transformation_cuts_enabled)
    {
        cancel_stft_slice_view();
        try
        {
            // Wait for refresh to be enabled for notify
            while (parent_->holovibes_.get_compute_pipe()->get_refresh_request())
                continue;
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what();
        }
        parent_->cd_.set_time_transformation_cuts_enabled(false);
    }
    parent_->notify();
}

void ViewPanel::set_auto_contrast_cuts()
{
    if (auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

void ViewPanel::set_fft_shift(const bool value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_fft_shift_enabled(value);
    parent_->pipe_refresh();
}

void ViewPanel::update_lens_view(bool value)
{
    parent_->cd_.set_gpu_lens_display_enabled(value);

    if (value)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = parent_->mainDisplay->framePosition() + QPoint(parent_->mainDisplay->width() + 310, 0);
            ICompute* pipe = parent_->holovibes_.get_compute_pipe().get();

            const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_input_queue()->get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, parent_->auxiliary_window_max_size);

            parent_->lens_window.reset(new RawWindow(pos,
                                                     QSize(lens_window_width, lens_window_height),
                                                     pipe->get_lens_queue().get(),
                                                     KindOfView::Lens));

            parent_->lens_window->setTitle("Lens view");
            parent_->lens_window->setCd(&(parent_->cd_));

            // when the window is destoryed, disable_lens_view() will be triggered
            connect(parent_->lens_window.get(), SIGNAL(destroyed()), parent_, SLOT(disable_lens_view()));
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }

    else
    {
        disable_lens_view();
        parent_->lens_window.reset(nullptr);
    }

    parent_->pipe_refresh();
}

void ViewPanel::disable_lens_view()
{
    if (parent_->lens_window)
        disconnect(parent_->lens_window.get(), SIGNAL(destroyed()), parent_, SLOT(disable_lens_view()));

    parent_->cd_.set_gpu_lens_display_enabled(false);
    parent_->holovibes_.get_compute_pipe()->request_disable_lens_view();
    parent_->notify();
}

void ViewPanel::update_raw_view(bool value)
{
    if (value)
    {
        if (parent_->cd_.batch_size > global::global_config.output_queue_max_size)
        {
            parent_->ui.RawDisplayingCheckBox->setChecked(false);
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return;
        }

        auto pipe = parent_->holovibes_.get_compute_pipe();
        pipe->request_raw_view();

        // Wait for the raw view to be enabled for notify
        while (pipe->get_raw_view_requested())
            continue;

        const camera::FrameDescriptor& fd = parent_->holovibes_.get_gpu_input_queue()->get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, parent_->auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = parent_->mainDisplay->framePosition() + QPoint(parent_->mainDisplay->width() + 310, 0);
        parent_->raw_window.reset(
            new RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

        parent_->raw_window->setTitle("Raw view");
        parent_->raw_window->setCd(&(parent_->cd_));

        connect(parent_->raw_window.get(), SIGNAL(destroyed()), parent_, SLOT(disable_raw_view()));
    }
    else
    {
        parent_->raw_window.reset(nullptr);
        disable_raw_view();
    }
    parent_->pipe_refresh();
}

void ViewPanel::disable_raw_view()
{
    if (parent_->raw_window)
        disconnect(parent_->raw_window.get(), SIGNAL(destroyed()), parent_, SLOT(disable_raw_view()));

    auto pipe = parent_->holovibes_.get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;

    parent_->notify();
}

void ViewPanel::set_x_y()
{
    auto& fd = parent_->holovibes_.get_gpu_input_queue()->get_fd();
    uint x = parent_->ui.XSpinBox->value();
    uint y = parent_->ui.YSpinBox->value();

    if (x < fd.width)
        parent_->cd_.set_x_cuts(x);

    if (y < fd.height)
        parent_->cd_.set_y_cuts(y);
}

void ViewPanel::set_x_accu()
{
    parent_->cd_.set_x_accu(parent_->ui.XAccuCheckBox->isChecked(), parent_->ui.XAccSpinBox->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_y_accu()
{
    parent_->cd_.set_y_accu(parent_->ui.YAccuCheckBox->isChecked(), parent_->ui.YAccSpinBox->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_p(int value)
{
    if (parent_->is_raw_mode())
        return;

    if (value < static_cast<int>(parent_->cd_.time_transformation_size))
    {
        parent_->cd_.pindex = value;
        parent_->pipe_refresh();
        parent_->notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void ViewPanel::increment_p()
{
    if (parent_->is_raw_mode())
        return;

    if (parent_->cd_.pindex < parent_->cd_.time_transformation_size)
    {
        parent_->cd_.set_pindex(parent_->cd_.pindex + 1);
        set_auto_contrast();
        parent_->notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void ViewPanel::decrement_p()
{
    if (parent_->is_raw_mode())
        return;

    if (parent_->cd_.pindex > 0)
    {
        parent_->cd_.set_pindex(parent_->cd_.pindex - 1);
        set_auto_contrast();
        parent_->notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void ViewPanel::set_p_accu()
{
    parent_->cd_.set_p_accu(parent_->ui.PAccuCheckBox->isChecked(), parent_->ui.PAccSpinBox->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_q(int value)
{
    parent_->cd_.set_q_index(value);
    parent_->notify();
}

void ViewPanel::set_q_acc()
{
    parent_->cd_.set_q_accu(parent_->ui.Q_AccuCheckBox->isChecked(), parent_->ui.Q_AccSpinBox->value());
    parent_->notify();
}

void ViewPanel::rotateTexture()
{
    WindowKind curWin = parent_->cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        parent_->displayAngle = (parent_->displayAngle == 270.f) ? 0.f : parent_->displayAngle + 90.f;
        parent_->mainDisplay->setAngle(parent_->displayAngle);
    }
    else if (parent_->sliceXZ && curWin == WindowKind::XZview)
    {
        parent_->xzAngle = (parent_->xzAngle == 270.f) ? 0.f : parent_->xzAngle + 90.f;
        parent_->sliceXZ->setAngle(parent_->xzAngle);
    }
    else if (parent_->sliceYZ && curWin == WindowKind::YZview)
    {
        parent_->yzAngle = (parent_->yzAngle == 270.f) ? 0.f : parent_->yzAngle + 90.f;
        parent_->sliceYZ->setAngle(parent_->yzAngle);
    }
    parent_->notify();
}

void ViewPanel::flipTexture()
{
    WindowKind curWin = parent_->cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        parent_->displayFlip = !parent_->displayFlip;
        parent_->mainDisplay->setFlip(parent_->displayFlip);
    }
    else if (parent_->sliceXZ && curWin == WindowKind::XZview)
    {
        parent_->xzFlip = !parent_->xzFlip;
        parent_->sliceXZ->setFlip(parent_->xzFlip);
    }
    else if (parent_->sliceYZ && curWin == WindowKind::YZview)
    {
        parent_->yzFlip = !parent_->yzFlip;
        parent_->sliceYZ->setFlip(parent_->yzFlip);
    }
    parent_->notify();
}

void ViewPanel::set_log_scale(const bool value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_log_scale_slice_enabled(parent_->cd_.current_window, value);
    if (value && parent_->cd_.contrast_enabled)
        set_auto_contrast();
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_accumulation(bool value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_accumulation(value);
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_accumulation_level(int value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->cd_.set_accumulation_level(value);
    parent_->pipe_refresh();
}

void ViewPanel::set_contrast_mode(bool value)
{
    if (parent_->is_raw_mode())
        return;

    parent_->change_window();
    parent_->cd_.set_contrast_mode(value);
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_auto_contrast()
{
    if (parent_->is_raw_mode())
        return;

    try
    {
        if (auto pipe = dynamic_cast<Pipe*>(parent_->holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(parent_->cd_.current_window);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }
}

void ViewPanel::set_auto_refresh_contrast(bool value)
{
    parent_->cd_.set_contrast_auto_refresh(value);
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::invert_contrast(bool value)
{
    if (!parent_->is_raw_mode() && parent_->cd_.set_contrast_invert(value))
    {
        parent_->pipe_refresh();
    }
}

void ViewPanel::set_contrast_min(const double value)
{
    if (parent_->is_raw_mode())
        return;

    if (parent_->cd_.contrast_enabled)
    {
        const float old_val = parent_->cd_.get_truncate_contrast_min();
        const float val = value;
        const float epsilon = 0.001f; // Precision in get_truncate_contrast_min is 2 decimals by default

        if (abs(old_val - val) > epsilon)
        {
            parent_->cd_.set_contrast_min(value);
            parent_->pipe_refresh();
        }
    }
}

void ViewPanel::set_contrast_max(const double value)
{
    if (parent_->is_raw_mode())
        return;

    if (parent_->cd_.contrast_enabled)
    {
        const float old_val = parent_->cd_.get_truncate_contrast_max();
        const float val = value;
        const float epsilon = 0.001f; // Precision in get_truncate_contrast_min is 2 decimals by default

        if (abs(old_val - val) > epsilon)
        {
            parent_->cd_.set_contrast_max(value);
            parent_->pipe_refresh();
        }
    }
}

void ViewPanel::toggle_renormalize(bool value)
{
    parent_->cd_.set_renorm_enabled(value);

    parent_->holovibes_.get_compute_pipe()->request_clear_img_acc();
    parent_->pipe_refresh();
}

void ViewPanel::display_reticle(bool value)
{
    parent_->cd_.set_reticle_enabled(value);
    if (value)
    {
        parent_->mainDisplay->getOverlayManager().create_overlay<Reticle>();
        parent_->mainDisplay->getOverlayManager().create_default();
    }
    else
    {
        parent_->mainDisplay->getOverlayManager().disable_all(Reticle);
    }
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::reticle_scale(double value)
{
    if (0 > value || value > 1)
        return;

    parent_->cd_.set_reticle_scale(value);
    parent_->pipe_refresh();
}
} // namespace holovibes::gui
