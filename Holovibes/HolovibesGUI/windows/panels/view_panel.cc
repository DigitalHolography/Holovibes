/*! \file
 *
 */

#include <limits>

#include "view_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

#include "view_struct.hh"

#include "API.hh"

#include "user_interface.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ViewPanel::ViewPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().view_panel = this;
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

// TODO: use parameters instead of directly the GSH
void ViewPanel::view_callback(WindowKind, ViewWindow)
{
    const bool is_raw = api::get_compute_mode() == ComputeModeEnum::Raw;

    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_current_view().contrast.enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_current_view().contrast.auto_refresh);
    ui_->InvertContrastCheckBox->setChecked(api::get_current_view().contrast.invert);
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_current_view().get_contrast_min_logged());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_current_view().get_contrast_max_logged());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);
    window_selection->setCurrentIndex(static_cast<int>(api::get_current_view_kind()));
}

void ViewPanel::on_notify()
{
    const bool is_raw = api::get_compute_mode() == ComputeModeEnum::Raw;

    ui_->ViewModeComboBox->setCurrentIndex(static_cast<int>(api::get_image_type()));

    ui_->PhaseUnwrap2DCheckBox->setVisible(api::get_image_type() == ImageTypeEnum::PhaseIncrease ||
                                           api::get_image_type() == ImageTypeEnum::Argument);

    ui_->TimeTransformationCutsCheckBox->setEnabled(ui_->timeTransformationSizeSpinBox->value() >=
                                                    MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);
    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && api::get_cuts_view_enabled());

    ui_->FFTShiftCheckBox->setChecked(api::get_fft_shift_enabled());
    ui_->FFTShiftCheckBox->setEnabled(true);

    ui_->LensViewCheckBox->setChecked(api::get_lens_view_enabled());

    ui_->RawDisplayingCheckBox->setHidden(false);
    ui_->RawDisplayingCheckBox->setEnabled(!is_raw);
    ui_->RawDisplayingCheckBox->setChecked(!is_raw && api::get_raw_view_enabled());

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && api::get_current_view().contrast.enabled);
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api::get_current_view().contrast.auto_refresh);
    ui_->InvertContrastCheckBox->setChecked(api::get_current_view().contrast.invert);
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMinDoubleSpinBox->setValue(api::get_current_view().get_contrast_min_logged());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api::get_current_view().contrast.auto_refresh);
    ui_->ContrastMaxDoubleSpinBox->setValue(api::get_current_view().get_contrast_max_logged());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);
    window_selection->setCurrentIndex(static_cast<int>(api::get_current_view_kind()));

    // Log
    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw && api::get_current_view().log_enabled);

    // ImgAccWindow
    auto set_xyzf_visibility = [&](bool val)
    {
        ui_->ImgAccuLabel->setVisible(val);
        ui_->ImgAccuSpinBox->setVisible(val);
        ui_->RotatePushButton->setVisible(val);
        ui_->FlipPushButton->setVisible(val);
    };

    if (api::get_current_view_kind() == WindowKind::ViewFilter2D)
        set_xyzf_visibility(false);
    else
    {
        set_xyzf_visibility(true);

        ui_->ImgAccuSpinBox->setValue(api::get_current_view_as_view_xyz().output_image_accumulation);

        ui_->RotatePushButton->setText(
            ("Rot " + std::to_string(static_cast<int>(api::get_current_view_as_view_xyz().rotation))).c_str());
        ui_->FlipPushButton->setText(
            ("Flip " + std::to_string(api::get_current_view_as_view_xyz().horizontal_flip)).c_str());
    }

    // Deactivate previous maximum (chetor)
    ui_->PSpinBox->setMaximum(INT_MAX);
    ui_->PAccSpinBox->setMaximum(INT_MAX);

    // p accu

    ui_->PAccSpinBox->setValue(api::get_view_accu_p().width);
    ui_->PSpinBox->setValue(api::get_view_accu_p().start);
    ui_->PAccSpinBox->setEnabled(api::get_image_type() != ImageTypeEnum::PhaseIncrease);

    // Enforce maximum value for p_index and p_accu_level
    ui_->PSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_p().width - 1);
    ui_->PAccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_p().start - 1);
    ui_->PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = api::get_time_transformation() == TimeTransformationEnum::SSA_STFT;
    ui_->Q_AccSpinBox->setVisible(is_ssa_stft && !is_raw);
    ui_->Q_SpinBox->setVisible(is_ssa_stft && !is_raw);
    ui_->Q_Label->setVisible(is_ssa_stft && !is_raw);
    ui_->QaccLabel->setVisible(is_ssa_stft && !is_raw);

    // Deactivate previous maximum (chetor + unused)
    ui_->Q_SpinBox->setMaximum(INT_MAX);
    ui_->Q_AccSpinBox->setMaximum(INT_MAX);

    ui_->Q_AccSpinBox->setValue(api::get_view_accu_q().width);
    ui_->Q_SpinBox->setValue(api::get_view_accu_q().start);

    ui_->Q_SpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_q().width - 1);
    ui_->Q_AccSpinBox->setMaximum(api::get_time_transformation_size() - api::get_view_accu_q().start - 1);

    // XY accu
    ui_->XAccSpinBox->setValue(api::get_view_accu_x().width);
    ui_->YAccSpinBox->setValue(api::get_view_accu_y().width);

    int max_width = 0;
    int max_height = 0;
    if (api::get_gpu_input_queue_ptr() != nullptr)
    {
        max_width = api::get_gpu_input_queue().get_fd().width - 1;
        max_height = api::get_gpu_input_queue().get_fd().height - 1;
    }
    else
    {
        api::change_view_accu_x()->start = 0;
        api::change_view_accu_y()->start = 0;
    }

    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, api::get_view_accu_x().start);
    QSpinBoxQuietSetValue(ui_->YSpinBox, api::get_view_accu_y().start);

    ui_->RenormalizeCheckBox->setChecked(api::get_renorm_enabled());
    ui_->ReticleScaleDoubleSpinBox->setEnabled(api::get_reticle().display_enabled);
    ui_->ReticleScaleDoubleSpinBox->setValue(api::get_reticle().scale);
    ui_->DisplayReticleCheckBox->setChecked(api::get_reticle().display_enabled);
}

void ViewPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, isHidden(), "panels", "view hidden", isHidden());
    ui_->actionView->setChecked(!h);
    setHidden(h);

    time_transformation_cuts_window_max_size =
        json_get_or_default(j_us, 512, "windows", "time transformation cuts window max size");
}

void ViewPanel::save_gui(json& j_us)
{
    j_us["panels"]["view hidden"] = isHidden();
    j_us["windows"]["time transformation cuts window max size"] = time_transformation_cuts_window_max_size;
}

void ViewPanel::set_view_mode(const QString& value) { parent_->set_view_image_type(value); }

void ViewPanel::set_unwrapping_2d(const bool value) { api::detail::set_value<Unwrap2DRequested>(value); }

void ViewPanel::update_3d_cuts_view(bool checked) { api::detail::set_value<CutsViewEnabled>(checked); }

void ViewPanel::cancel_time_transformation_cuts() { api::detail::set_value<TimeTransformationCutsEnable>(false); }

void ViewPanel::set_auto_contrast_cuts()
{
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewXZ);
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewYZ);
}

void ViewPanel::set_fft_shift(const bool value) { api::set_fft_shift_enabled(value); }

void ViewPanel::update_lens_view(bool checked) { api::set_lens_view_enabled(checked); }

void ViewPanel::update_raw_view(bool checked) { api::set_raw_view_enabled(checked); }

void ViewPanel::close_windows() {}

void ViewPanel::set_x_y()
{
    api::detail::change_value<ViewAccuX>()->start = ui_->XSpinBox->value();
    api::detail::change_value<ViewAccuY>()->start = ui_->YSpinBox->value();
}

void ViewPanel::set_x_accu() { api::change_view_accu_x()->width = ui_->XAccSpinBox->value(); }

void ViewPanel::set_y_accu() { api::change_view_accu_y()->width = ui_->YAccSpinBox->value(); }

void ViewPanel::set_p(int value) { api::change_view_accu_p()->start = value; }

void ViewPanel::increment_p() { set_p(api::get_view_accu_p().start + 1); }

void ViewPanel::decrement_p() { set_p(api::get_view_accu_p().start - 1); }

void ViewPanel::set_p_accu() { api::change_view_accu_p()->width = ui_->PAccSpinBox->value(); }

void ViewPanel::set_q(int value) { api::change_view_accu_q()->start = value; }

void ViewPanel::set_q_acc() { api::change_view_accu_q()->width = ui_->Q_AccSpinBox->value(); }

void ViewPanel::rotateTexture()
{
    double rotation = api::get_current_view_as_view_xyz().rotation;
    double new_rot = (rotation == 270.f) ? 0.f : rotation + 90.f;
    api::change_current_view_as_view_xyz()->rotation = new_rot;

    if (api::get_current_view_kind() == WindowKind::ViewXY)
        UserInterface::instance().xy_window->setTransform();
    else if (UserInterface::instance().sliceXZ && api::get_current_view_kind() == WindowKind::ViewXZ)
        UserInterface::instance().sliceXZ->setTransform();
    else if (UserInterface::instance().sliceYZ && api::get_current_view_kind() == WindowKind::ViewYZ)
        UserInterface::instance().sliceYZ->setTransform();
}

void ViewPanel::flipTexture()
{
    api::change_current_view_as_view_xyz()->horizontal_flip = !api::get_current_view_as_view_xyz().horizontal_flip;

    // FIXME API => TRIGGER
    if (api::get_current_view_kind() == WindowKind::ViewXY)
        UserInterface::instance().xy_window->setTransform();
    else if (UserInterface::instance().sliceXZ && api::get_current_view_kind() == WindowKind::ViewXZ)
        UserInterface::instance().sliceXZ->setTransform();
    else if (UserInterface::instance().sliceYZ && api::get_current_view_kind() == WindowKind::ViewYZ)
        UserInterface::instance().sliceYZ->setTransform();
}

void ViewPanel::set_log_scale(const bool value) { api::change_current_view()->log_enabled = value; }

void ViewPanel::set_accumulation_level(int value)
{
    api::change_current_view_as_view_xyz()->output_image_accumulation = value;
}

void ViewPanel::set_contrast_mode(bool value) { api::change_current_view()->contrast.enabled = value; }

void ViewPanel::request_exec_contrast_current_window() { api::request_exec_contrast_current_window(); }

void ViewPanel::set_auto_refresh_contrast(bool value) { api::change_current_view()->contrast.auto_refresh = value; }

void ViewPanel::invert_contrast(bool value) { api::change_current_view()->contrast.invert = value; }

void ViewPanel::set_contrast_min(const double value) { api::set_current_window_contrast_min(value); }

void ViewPanel::set_contrast_max(const double value) { api::set_current_window_contrast_max(value); }

void ViewPanel::toggle_renormalize(bool value)
{
    api::set_renorm_enabled(value);

    if (api::get_import_type() != ImportTypeEnum::None)
    {
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewXY);
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewXZ);
        api::get_compute_pipe().get_rendering().request_view_clear_image_accumulation(WindowKind::ViewYZ);
    }
}

void ViewPanel::display_reticle(bool value) { api::change_reticle()->display_enabled = value; }

void ViewPanel::reticle_scale(double value) { api::change_reticle()->scale = value; }
} // namespace holovibes::gui
