#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"
#include "API.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
{
    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    p_left_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));
}

ViewPanel::~ViewPanel()
{
    delete p_left_shortcut_;
    delete p_right_shortcut_;
}

void ViewPanel::on_notify()
{
    const bool is_raw = api::is_raw_mode();

    ui_->ViewModeComboBox->setCurrentIndex(static_cast<int>(api::get_img_type()));

    ui_->PhaseUnwrap2DCheckBox->setEnabled(api::get_cd().img_type == ImgType::PhaseIncrease || api::get_cd().img_type == ImgType::Argument);
    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && api::get_cd().time_transformation_cuts_enabled);
    ui_->TimeTransformationCutsCheckBox->setEnabled(ui_->timeTransformationSizeSpinBox->value() >=
                                                    MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);
    ui_->FFTShiftCheckBox->setChecked(api::get_cd().fft_shift_enabled);
    ui_->FFTShiftCheckBox->setEnabled(true);
    ui_->LensViewCheckBox->setChecked(api::get_cd().gpu_lens_display_enabled);
    ui_->RawDisplayingCheckBox->setEnabled(!is_raw);
    ui_->RawDisplayingCheckBox->setChecked(!is_raw && api::get_cd().raw_view_enabled);

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_cd().contrast_enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_cd().contrast_auto_refresh);

    // Contrast Spinbox
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_cd().contrast_auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_cd().get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_cd().contrast_auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_cd().get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(api::get_cd().time_transformation_cuts_enabled);
    window_selection->setCurrentIndex(window_selection->isEnabled() ? static_cast<int>(api::get_cd().current_window.load()) : 0);

    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw && api::get_cd().get_img_log_scale_slice_enabled(api::get_cd().current_window.load()));
    ui_->ImgAccuCheckBox->setEnabled(true);
    ui_->ImgAccuCheckBox->setChecked(!is_raw && api::get_cd().get_img_acc_slice_enabled(api::get_cd().current_window.load()));
    ui_->ImgAccuSpinBox->setValue(api::get_cd().get_img_acc_slice_level(api::get_cd().current_window.load()));
    if (api::get_cd().current_window == WindowKind::XYview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().displayAngle))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().displayFlip)).c_str());
    }
    else if (api::get_cd().current_window == WindowKind::XZview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().xzAngle))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().xzFlip)).c_str());
    }
    else if (api::get_cd().current_window == WindowKind::YZview)
    {
        ui_->RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().yzAngle))).c_str());
        ui_->FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().yzFlip)).c_str());
    }

    // p accu
    ui_->PAccuCheckBox->setEnabled(api::get_cd().img_type != ImgType::PhaseIncrease);
    ui_->PAccuCheckBox->setChecked(api::get_cd().p_accu_enabled);
    ui_->PAccSpinBox->setMaximum(api::get_cd().time_transformation_size - 1);

    api::get_cd().check_p_limits();
    ui_->PAccSpinBox->setValue(api::get_cd().p_acc_level);
    ui_->PSpinBox->setValue(api::get_cd().pindex);
    ui_->PAccSpinBox->setEnabled(api::get_cd().img_type != ImgType::PhaseIncrease);
    if (api::get_cd().p_accu_enabled)
    {
        ui_->PSpinBox->setMaximum(api::get_cd().time_transformation_size - api::get_cd().p_acc_level - 1);
        ui_->PAccSpinBox->setMaximum(api::get_cd().time_transformation_size - api::get_cd().pindex - 1);
    }
    else
    {
        ui_->PSpinBox->setMaximum(api::get_cd().time_transformation_size - 1);
    }
    ui_->PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = api::get_cd().time_transformation == TimeTransformation::SSA_STFT;
    ui_->Q_AccuCheckBox->setEnabled(is_ssa_stft && !is_raw);
    ui_->Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui_->Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui_->Q_AccuCheckBox->setChecked(api::get_cd().q_acc_enabled);
    ui_->Q_AccSpinBox->setMaximum(api::get_cd().time_transformation_size - 1);

    api::get_cd().check_q_limits();
    ui_->Q_AccSpinBox->setValue(api::get_cd().q_acc_level);
    ui_->Q_SpinBox->setValue(api::get_cd().q_index);
    if (api::get_cd().q_acc_enabled)
    {
        ui_->Q_SpinBox->setMaximum(api::get_cd().time_transformation_size - api::get_cd().q_acc_level - 1);
        ui_->Q_AccSpinBox->setMaximum(api::get_cd().time_transformation_size - api::get_cd().q_index - 1);
    }
    else
    {
        ui_->Q_SpinBox->setMaximum(api::get_cd().time_transformation_size - 1);
    }

    // XY accu
    ui_->XAccuCheckBox->setChecked(api::get_cd().x_accu_enabled);
    ui_->XAccSpinBox->setValue(api::get_cd().x_acc_level);
    ui_->YAccuCheckBox->setChecked(api::get_cd().y_accu_enabled);
    ui_->YAccSpinBox->setValue(api::get_cd().y_acc_level);

    int max_width = 0;
    int max_height = 0;
    if (api::get_gpu_input_queue() != nullptr)
    {
        max_width = api::get_gpu_input_queue_fd_width() - 1;
        max_height = api::get_gpu_input_queue_fd_height() - 1;
    }
    else
    {
        api::get_cd().x_cuts = 0;
        api::get_cd().y_cuts = 0;
    }
    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, api::get_cd().x_cuts);
    QSpinBoxQuietSetValue(ui_->YSpinBox, api::get_cd().y_cuts);

    ui_->RenormalizeCheckBox->setChecked(api::get_cd().renorm_enabled);
    ui_->ReticleScaleDoubleSpinBox->setEnabled(api::get_cd().reticle_enabled);
    ui_->ReticleScaleDoubleSpinBox->setValue(api::get_cd().reticle_scale);
    ui_->DisplayReticleCheckBox->setChecked(api::get_cd().reticle_enabled);
}

void ViewPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    ui_->actionView->setChecked(!ptree.get<bool>("view.hidden", isHidden()));

    // UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size_ =
    //     ptree.get<uint>("display.time_transformation_cuts_window_max_size", UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size_);
    // UserInterfaceDescriptor::instance().displayAngle = ptree.get("view.mainWindow_rotate", UserInterfaceDescriptor::instance().displayAngle);
    // UserInterfaceDescriptor::instance().xzAngle_ = ptree.get<float>("view.xCut_rotate", UserInterfaceDescriptor::instance().xzAngle_);
    // UserInterfaceDescriptor::instance().yzAngle_ = ptree.get<float>("view.yCut_rotate", UserInterfaceDescriptor::instance().yzAngle_);
    // UserInterfaceDescriptor::instance().displayFlip = ptree.get("view.mainWindow_flip", UserInterfaceDescriptor::instance().displayFlip);
    // UserInterfaceDescriptor::instance().xzFlip_ = ptree.get("view.xCut_flip", UserInterfaceDescriptor::instance().xzFlip_);
    // UserInterfaceDescriptor::instance().yzFlip_ = ptree.get("view.yCut_flip", UserInterfaceDescriptor::instance().yzFlip_);
}

void ViewPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<bool>("view.hidden", isHidden());

    // ptree.put<uint>("display.time_transformation_cuts_window_max_size", UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size_);
    // ptree.put<float>("view.mainWindow_rotate", UserInterfaceDescriptor::instance().displayAngle);
    // ptree.put<float>("view.xCut_rotate", UserInterfaceDescriptor::instance().xzAngle_);
    // ptree.put<float>("view.yCut_rotate", UserInterfaceDescriptor::instance().yzAngle_);
    // ptree.put<int>("view.mainWindow_flip", UserInterfaceDescriptor::instance().displayFlip);
    // ptree.put<int>("view.xCut_flip", UserInterfaceDescriptor::instance().xzFlip_);
    // ptree.put<int>("view.yCut_flip", UserInterfaceDescriptor::instance().yzFlip_);
}

void ViewPanel::set_view_mode(const QString& value) { parent_->set_view_image_type(value); }

void ViewPanel::set_unwrapping_2d(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_unwrapping_2d(value);

    parent_->notify();
}

void ViewPanel::toggle_time_transformation_cuts(bool checked)
{

    QComboBox* winSelection = ui_->WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());

    if (!checked)
    {
        cancel_time_transformation_cuts();
        return;
    }

    const bool res = api::toggle_time_transformation_cuts(*parent_);

    if (res)
    {
        set_auto_contrast_cuts();
        parent_->notify();
    }
    else
    {
        cancel_time_transformation_cuts();
    }
}

void ViewPanel::cancel_time_transformation_cuts()
{
    if (!api::get_time_transformation_cuts_enabled())
        return;

    std::function<void()> callback = []() { return; };

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        callback = ([=]() {
            api::set_time_transformation_cuts_enabled(false);
            pipe->delete_stft_slice_queue();

            ui_->TimeTransformationCutsCheckBox->setChecked(false);
            parent_->notify();
        });
    }

    api::cancel_time_transformation_cuts(callback);

    parent_->notify();
}

void ViewPanel::set_auto_contrast_cuts()
{
    api::set_auto_contrast_cuts();
}

void ViewPanel::set_fft_shift(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_fft_shift(value);

    api::pipe_refresh();
}

void ViewPanel::update_lens_view(bool value)
{
    api::set_gpu_lens_display_enabled(value);

    if (value)
    {
        const bool res = api::set_lens_view();

        if (res)
        {
            connect(UserInterfaceDescriptor::instance().lens_window.get(),
                    SIGNAL(destroyed()),
                    this,
                    SLOT(disable_lens_view()));
        }
    }
    else
    {
        disable_lens_view();
    }

    api::pipe_refresh();
}

void ViewPanel::disable_lens_view()
{
    if (UserInterfaceDescriptor::instance().lens_window)
        disconnect(UserInterfaceDescriptor::instance().lens_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_lens_view()));

    api::disable_lens_view();
    parent_->notify();
}

void ViewPanel::update_raw_view(bool value)
{

    if (value)
    {
        if (api::get_batch_size() > global::global_config.output_queue_max_size)
        {
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return;
        }

        api::set_raw_view();
        connect(api::get_raw_window().get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
    }
    else
    {
        disable_raw_view();
    }

}
void ViewPanel::disable_raw_view()
{
    if (UserInterfaceDescriptor::instance().raw_window)
        disconnect(UserInterfaceDescriptor::instance().raw_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_raw_view()));

    api::disable_raw_view();

    parent_->notify();
}

void ViewPanel::set_x_y()
{
    api::set_x_y(ui_->XSpinBox->value(), ui_->YSpinBox->value());
}

void ViewPanel::set_x_accu()
{
    api::set_x_accu(ui_->XAccuCheckBox->isChecked(), ui_->XAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_y_accu()
{
    api::set_y_accu(ui_->YAccuCheckBox->isChecked(), ui_->YAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_p(int value)
{
    if (api::is_raw_mode())
        return;

    if (value >= static_cast<int>(api::get_time_transformation_size()))
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    api::set_p(value);

    parent_->notify();
}

void ViewPanel::increment_p()
{
    LOG_FUNC;

    if (api::is_raw_mode())
        return;

    if (api::get_pindex() >= api::get_time_transformation_size())
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    api::increment_p();

    set_auto_contrast();
    parent_->notify();
}

void ViewPanel::decrement_p()
{
    if (api::is_raw_mode())
        return;

    if (api::get_pindex() <= 0)
    {
        LOG_ERROR << "p param has to be between 1 and #img";
        return;
    }

    api::decrement_p();

    set_auto_contrast();
    parent_->notify();
}

void ViewPanel::set_p_accu()
{
    api::set_p_accu(ui_->PAccuCheckBox->isChecked(), ui_->PAccSpinBox->value());

    parent_->notify();
}

void ViewPanel::set_q(int value)
{
    api::set_q(value);
    
    parent_->notify();
}

void ViewPanel::set_q_acc()
{
    api::set_q_accu(ui_->Q_AccuCheckBox->isChecked(), ui_->Q_AccSpinBox->value());
    
    parent_->notify();
}

void ViewPanel::rotateTexture()
{
    api::rotateTexture();

    parent_->notify();
}

void ViewPanel::flipTexture()
{
    api::flipTexture();

    parent_->notify();
}

void ViewPanel::set_log_scale(const bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_log_scale(value);

    parent_->notify();
}

void ViewPanel::set_accumulation(bool value)
{
    if (api::is_raw_mode())
        return;

    api::set_accumulation(value);

    parent_->notify();
}

void ViewPanel::set_accumulation_level(int value)
{
    if (api::is_raw_mode())
        return;

    api::set_accumulation_level(value);
}

void ViewPanel::set_contrast_mode(bool value)
{
    parent_->change_window();

    if (api::is_raw_mode())
        return;

    api::set_contrast_mode(value);

    parent_->notify();
}

void ViewPanel::set_auto_contrast()
{
    if (api::is_raw_mode())
        return;

    api::set_auto_contrast();
}

void ViewPanel::set_auto_refresh_contrast(bool value)
{
    api::set_auto_refresh_contrast(value);

    parent_->notify();
}

void ViewPanel::invert_contrast(bool value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::invert_contrast(value);
}

void ViewPanel::set_contrast_min(const double value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::set_contrast_min(value);
}

void ViewPanel::set_contrast_max(const double value)
{
    if (api::is_raw_mode())
        return;

    if (!api::get_contrast_enabled())
        return;

    api::set_contrast_max(value);
}

void ViewPanel::toggle_renormalize(bool value)
{
    api::toggle_renormalize(value);
}

void ViewPanel::display_reticle(bool value)
{
    api::display_reticle(value);

    parent_->notify();
}

void ViewPanel::reticle_scale(double value)
{
    if (0 > value || value > 1)
        return;

    api::reticle_scale(value);
}
} // namespace holovibes::gui
