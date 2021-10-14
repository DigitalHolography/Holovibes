#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
{
}

ViewPanel::~ViewPanel() {}

void ViewPanel::on_notify()
{
    const bool is_raw = parent_->is_raw_mode();

    ui_->PhaseUnwrap2DCheckBox->setEnabled(parent_->cd_.img_type == ImgType::PhaseIncrease ||
                                           parent_->cd_.img_type == ImgType::Argument);
    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && parent_->cd_.time_transformation_cuts_enabled);
    ui_->TimeTransformationCutsCheckBox->setEnabled(ui_->timeTransformationSizeSpinBox->value() >=
                                                    MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);
    ui_->FFTShiftCheckBox->setChecked(parent_->cd_.fft_shift_enabled);
    ui_->FFTShiftCheckBox->setEnabled(true);
    ui_->LensViewCheckBox->setChecked(parent_->cd_.gpu_lens_display_enabled);
    ui_->RawDisplayingCheckBox->setEnabled(!is_raw);
    ui_->RawDisplayingCheckBox->setChecked(!is_raw && parent_->cd_.raw_view_enabled);

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && parent_->cd_.contrast_enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(parent_->cd_.contrast_auto_refresh);

    // Contrast Spinbox
    ui_->ContrastMinDoubleSpinBox->setEnabled(!parent_->cd_.contrast_auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(parent_->cd_.get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!parent_->cd_.contrast_auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(parent_->cd_.get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(parent_->cd_.time_transformation_cuts_enabled);
    window_selection->setCurrentIndex(
        window_selection->isEnabled() ? static_cast<int>(parent_->cd_.current_window.load()) : 0);

    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw &&
                                      parent_->cd_.get_img_log_scale_slice_enabled(parent_->cd_.current_window.load()));
    ui_->ImgAccuCheckBox->setEnabled(true);
    ui_->ImgAccuCheckBox->setChecked(!is_raw &&
                                     parent_->cd_.get_img_acc_slice_enabled(parent_->cd_.current_window.load()));
    ui_->ImgAccuSpinBox->setValue(parent_->cd_.get_img_acc_slice_level(parent_->cd_.current_window.load()));
    if (parent_->cd_.current_window == WindowKind::XYview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(displayFlip)).c_str());
    }
    else if (parent_->cd_.current_window == WindowKind::XZview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(xzAngle_))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(xzFlip_)).c_str());
    }
    else if (parent_->cd_.current_window == WindowKind::YZview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(yzAngle_))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(yzFlip_)).c_str());
    }

    // p accu
    ui_->PAccuCheckBox->setEnabled(parent_->cd_.img_type != ImgType::PhaseIncrease);
    ui_->PAccuCheckBox->setChecked(parent_->cd_.p_accu_enabled);
    ui_->PAccSpinBox->setMaximum(parent_->cd_.time_transformation_size - 1);

    parent_->cd_.check_p_limits();
    ui_->PAccSpinBox->setValue(parent_->cd_.p_acc_level);
    ui_->PSpinBox->setValue(parent_->cd_.pindex);
    ui_->PAccSpinBox->setEnabled(parent_->cd_.img_type != ImgType::PhaseIncrease);
    if (parent_->cd_.p_accu_enabled)
    {
        ui_->PSpinBox->setMaximum(parent_->cd_.time_transformation_size - parent_->cd_.p_acc_level - 1);
        ui_->PAccSpinBox->setMaximum(parent_->cd_.time_transformation_size - parent_->cd_.pindex - 1);
    }
    else
    {
        ui_->PSpinBox->setMaximum(parent_->cd_.time_transformation_size - 1);
    }
    ui_->PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = parent_->cd_.time_transformation == TimeTransformation::SSA_STFT;
    ui_->Q_AccuCheckBox->setEnabled(is_ssa_stft && !is_raw);
    ui_->Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui_->Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui_->Q_AccuCheckBox->setChecked(parent_->cd_.q_acc_enabled);
    ui_->Q_AccSpinBox->setMaximum(parent_->cd_.time_transformation_size - 1);

    parent_->cd_.check_q_limits();
    ui_->Q_AccSpinBox->setValue(parent_->cd_.q_acc_level);
    ui_->Q_SpinBox->setValue(parent_->cd_.q_index);
    if (parent_->cd_.q_acc_enabled)
    {
        ui_->Q_SpinBox->setMaximum(parent_->cd_.time_transformation_size - parent_->cd_.q_acc_level - 1);
        ui_->Q_AccSpinBox->setMaximum(parent_->cd_.time_transformation_size - parent_->cd_.q_index - 1);
    }
    else
    {
        ui_->Q_SpinBox->setMaximum(parent_->cd_.time_transformation_size - 1);
    }

    // XY accu
    ui_->XAccuCheckBox->setChecked(parent_->cd_.x_accu_enabled);
    ui_->XAccSpinBox->setValue(parent_->cd_.x_acc_level);
    ui_->YAccuCheckBox->setChecked(parent_->cd_.y_accu_enabled);
    ui_->YAccSpinBox->setValue(parent_->cd_.y_acc_level);

    int max_width = 0;
    int max_height = 0;
    if (parent_->holovibes_.get_gpu_input_queue() != nullptr)
    {
        max_width = parent_->holovibes_.get_gpu_input_queue()->get_fd().width - 1;
        max_height = parent_->holovibes_.get_gpu_input_queue()->get_fd().height - 1;
    }
    else
    {
        parent_->cd_.x_cuts = 0;
        parent_->cd_.y_cuts = 0;
    }
    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, parent_->cd_.x_cuts);
    QSpinBoxQuietSetValue(ui_->YSpinBox, parent_->cd_.y_cuts);

    ui_->RenormalizeCheckBox->setChecked(parent_->cd_.renorm_enabled);
    ui_->ReticleScaleDoubleSpinBox->setEnabled(parent_->cd_.reticle_enabled);
    ui_->ReticleScaleDoubleSpinBox->setValue(parent_->cd_.reticle_scale);
    ui_->DisplayReticleCheckBox->setChecked(parent_->cd_.reticle_enabled);
}

void ViewPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    time_transformation_cuts_window_max_size_ =
        ptree.get<uint>("display.time_transformation_cuts_window_max_size", time_transformation_cuts_window_max_size_);

    displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
    xzAngle_ = ptree.get<float>("view.xCut_rotate", xzAngle_);
    yzAngle_ = ptree.get<float>("view.yCut_rotate", yzAngle_);
    displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
    xzFlip_ = ptree.get("view.xCut_flip", xzFlip_);
    yzFlip_ = ptree.get("view.yCut_flip", yzFlip_);
}

void ViewPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<uint>("display.time_transformation_cuts_window_max_size", time_transformation_cuts_window_max_size_);

    ptree.put<float>("view.mainWindow_rotate", displayAngle);
    ptree.put<float>("view.xCut_rotate", xzAngle_);
    ptree.put<float>("view.yCut_rotate", yzAngle_);
    ptree.put<int>("view.mainWindow_flip", displayFlip);
    ptree.put<int>("view.xCut_flip", xzFlip_);
    ptree.put<int>("view.yCut_flip", yzFlip_);
}

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
    QComboBox* winSelection = ui_->WindowSelectionComboBox;
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

            if (time_transformation_size > time_transformation_cuts_window_max_size_)
                time_transformation_size = time_transformation_cuts_window_max_size_;

            while (parent_->holovibes_.get_compute_pipe()->get_update_time_transformation_size_request())
                continue;
            while (parent_->holovibes_.get_compute_pipe()->get_cuts_request())
                continue;
            sliceXZ.reset(new SliceWindow(xzPos,
                                          QSize(parent_->mainDisplay->width(), time_transformation_size),
                                          parent_->holovibes_.get_compute_pipe()->get_stft_slice_queue(0).get(),
                                          KindOfView::SliceXZ,
                                          parent_));
            sliceXZ->setTitle("XZ view");
            sliceXZ->setAngle(xzAngle_);
            sliceXZ->setFlip(xzFlip_);
            sliceXZ->setCd(&(parent_->cd_));

            sliceYZ.reset(new SliceWindow(yzPos,
                                          QSize(time_transformation_size, parent_->mainDisplay->height()),
                                          parent_->holovibes_.get_compute_pipe()->get_stft_slice_queue(1).get(),
                                          KindOfView::SliceYZ,
                                          parent_));
            sliceYZ->setTitle("YZ view");
            sliceYZ->setAngle(yzAngle_);
            sliceYZ->setFlip(yzFlip_);
            sliceYZ->setCd(&(parent_->cd_));

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
    sliceXZ.reset(nullptr);
    sliceYZ.reset(nullptr);

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

            ui_->TimeTransformationCutsCheckBox->setChecked(false);
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

            lens_window.reset(new RawWindow(pos,
                                            QSize(lens_window_width, lens_window_height),
                                            pipe->get_lens_queue().get(),
                                            KindOfView::Lens));

            lens_window->setTitle("Lens view");
            lens_window->setCd(&(parent_->cd_));

            // when the window is destoryed, disable_lens_view() will be triggered
            connect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }

    else
    {
        disable_lens_view();
        lens_window.reset(nullptr);
    }

    parent_->pipe_refresh();
}

void ViewPanel::disable_lens_view()
{
    if (lens_window)
        disconnect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));

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
            ui_->RawDisplayingCheckBox->setChecked(false);
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
        raw_window.reset(
            new RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

        raw_window->setTitle("Raw view");
        raw_window->setCd(&(parent_->cd_));

        connect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
    }
    else
    {
        raw_window.reset(nullptr);
        disable_raw_view();
    }
    parent_->pipe_refresh();
}

void ViewPanel::disable_raw_view()
{
    if (raw_window)
        disconnect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));

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
    uint x = ui_->XSpinBox->value();
    uint y = ui_->YSpinBox->value();

    if (x < fd.width)
        parent_->cd_.set_x_cuts(x);

    if (y < fd.height)
        parent_->cd_.set_y_cuts(y);
}

void ViewPanel::set_x_accu()
{
    parent_->cd_.set_x_accu(ui_->XAccuCheckBox->isChecked(), ui_->XAccSpinBox->value());
    parent_->pipe_refresh();
    parent_->notify();
}

void ViewPanel::set_y_accu()
{
    parent_->cd_.set_y_accu(ui_->YAccuCheckBox->isChecked(), ui_->YAccSpinBox->value());
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
    parent_->cd_.set_p_accu(ui_->PAccuCheckBox->isChecked(), ui_->PAccSpinBox->value());
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
    parent_->cd_.set_q_accu(ui_->Q_AccuCheckBox->isChecked(), ui_->Q_AccSpinBox->value());
    parent_->notify();
}

void ViewPanel::rotateTexture()
{
    WindowKind curWin = parent_->cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        displayAngle = (displayAngle == 270.f) ? 0.f : displayAngle + 90.f;
        parent_->mainDisplay->setAngle(displayAngle);
    }
    else if (sliceXZ && curWin == WindowKind::XZview)
    {
        xzAngle_ = (xzAngle_ == 270.f) ? 0.f : xzAngle_ + 90.f;
        sliceXZ->setAngle(xzAngle_);
    }
    else if (sliceYZ && curWin == WindowKind::YZview)
    {
        yzAngle_ = (yzAngle_ == 270.f) ? 0.f : yzAngle_ + 90.f;
        sliceYZ->setAngle(yzAngle_);
    }
    parent_->notify();
}

void ViewPanel::flipTexture()
{
    WindowKind curWin = parent_->cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        displayFlip = !displayFlip;
        parent_->mainDisplay->setFlip(displayFlip);
    }
    else if (sliceXZ && curWin == WindowKind::XZview)
    {
        xzFlip_ = !xzFlip_;
        sliceXZ->setFlip(xzFlip_);
    }
    else if (sliceYZ && curWin == WindowKind::YZview)
    {
        yzFlip_ = !yzFlip_;
        sliceYZ->setFlip(yzFlip_);
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
