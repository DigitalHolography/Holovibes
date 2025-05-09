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
#include "GUI.hh"
#include "user_interface_descriptor.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

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

void ViewPanel::update_img_type(int img_type)
{
    ui_->ViewModeComboBox->setCurrentIndex(img_type);

    const int mom0 = static_cast<int>(ImgType::Moments_0);
    const int mom2 = static_cast<int>(ImgType::Moments_2);
    auto viewbox_view = qobject_cast<QListView*>(ui_->ViewModeComboBox->view());

    if (api_.input.get_data_type() == RecordedDataType::MOMENTS)
    {
        for (int i = 0; i < ui_->ViewModeComboBox->count(); i++)
        {
            if (i < mom0 || i > mom2)
                viewbox_view->setRowHidden(i, true); // Hide non-moments display options
        }

        if (img_type < mom0 || img_type > mom2)
            ui_->ViewModeComboBox->setCurrentIndex(mom0);
    }
    else
    {
        for (int i = 0; i < ui_->ViewModeComboBox->count(); i++)
            viewbox_view->setRowHidden(i, false); // Set all display options to be visible again
    }
}

void ViewPanel::on_notify()
{
    const bool is_raw = api_.compute.get_compute_mode() == Computation::Raw;
    const bool is_data_not_moments = !(api_.input.get_data_type() == RecordedDataType::MOMENTS);

    update_img_type(static_cast<int>(api_.compute.get_img_type()));

    ui_->PhaseUnwrap2DCheckBox->setVisible(api_.compute.get_img_type() == ImgType::PhaseIncrease ||
                                           api_.compute.get_img_type() == ImgType::Argument);
    ui_->PhaseUnwrap2DCheckBox->setChecked(api_.transform.get_unwrapping_2d());

    ui_->TimeTransformationCutsCheckBox->setChecked(!is_raw && api_.view.get_cuts_view_enabled());
    ui_->TimeTransformationCutsCheckBox->setEnabled(
        ui_->timeTransformationSizeSpinBox->value() >= MIN_IMG_NB_TIME_TRANSFORMATION_CUTS && is_data_not_moments);

    ui_->FFTShiftCheckBox->setChecked(api_.transform.get_fft_shift_enabled());
    ui_->FFTShiftCheckBox->setEnabled(true);

    ui_->RegistrationCheckBox->setChecked(api_.global_pp.get_registration_enabled());
    ui_->RegistrationCheckBox->setEnabled(true);
    ui_->RegistrationZoneSpinBox->setEnabled(api_.global_pp.get_registration_enabled());
    ui_->RegistrationZoneSpinBox->setValue(api_.global_pp.get_registration_zone());

    ui_->LensViewCheckBox->setChecked(api_.view.get_lens_view_enabled());
    ui_->LensViewCheckBox->setEnabled(is_data_not_moments);

    ui_->RawDisplayingCheckBox->setChecked(!is_raw && api_.view.get_raw_view_enabled());
    ui_->RawDisplayingCheckBox->setEnabled(!is_raw && is_data_not_moments);

    // Contrast
    ui_->ContrastCheckBox->setChecked(!is_raw && api_.contrast.get_contrast_enabled());
    ui_->ContrastCheckBox->setEnabled(true);
    ui_->AutoRefreshContrastCheckBox->setChecked(api_.contrast.get_contrast_auto_refresh());
    ui_->InvertContrastCheckBox->setChecked(api_.contrast.get_contrast_invert());
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api_.contrast.get_contrast_auto_refresh());
    ui_->ContrastMinDoubleSpinBox->setValue(api_.contrast.get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api_.contrast.get_contrast_auto_refresh());
    ui_->ContrastMaxDoubleSpinBox->setValue(api_.contrast.get_contrast_max());

    // Window selection
    QComboBox* window_selection = ui_->WindowSelectionComboBox;
    window_selection->setEnabled(!is_raw);

    // Display Reticle
    ui_->DisplayReticleDoubleSpinbox->setEnabled(ui_->DisplayReticleCheckbox->isChecked());
    ui_->DisplayReticleDoubleSpinbox->setValue(api_.contrast.get_reticle_scale());

    // Enable only row that are actually displayed on the screen
    QListView* window_selection_view = qobject_cast<QListView*>(window_selection->view());
    window_selection_view->setRowHidden(1, !api_.window_pp.get_enabled(WindowKind::XZview));
    window_selection_view->setRowHidden(2, !api_.window_pp.get_enabled(WindowKind::YZview));
    window_selection_view->setRowHidden(3, !api_.view.get_filter2d_view_enabled());

    // If one view gets disabled set to the standard XY view
    int index = static_cast<int>(api_.view.get_current_window_type());
    if (window_selection_view->isRowHidden(index))
    {
        index = 0;
        parent_->change_window(index);
    }

    window_selection->setCurrentIndex(index);

    // Log
    ui_->LogScaleCheckBox->setEnabled(true);
    ui_->LogScaleCheckBox->setChecked(!is_raw && api_.contrast.get_log_enabled());

    // ImgAccWindow
    auto set_xyzf_visibility = [&](bool val)
    {
        ui_->ImgAccuLabel->setVisible(val);
        ui_->ImgAccuSpinBox->setVisible(val);
        ui_->RotatePushButton->setVisible(val);
        ui_->FlipPushButton->setVisible(val);
    };

    if (api_.view.get_current_window_type() == WindowKind::Filter2D)
        set_xyzf_visibility(false);
    else
    {
        set_xyzf_visibility(true);

        ui_->ImgAccuSpinBox->setValue(api_.window_pp.get_accumulation_level());

        constexpr int max_digit_rotate = 3;
        constexpr int max_digit_flip = 1;

        std::string rotation_degree = std::to_string(static_cast<int>(api_.window_pp.get_rotation()));
        rotation_degree.insert(0, max_digit_rotate - rotation_degree.size(), ' ');

        auto current_rotation = ui_->RotatePushButton->text();
        current_rotation.replace(current_rotation.size() - 3, max_digit_rotate, rotation_degree.c_str());

        auto current_flip = ui_->FlipPushButton->text();
        current_flip.replace(current_flip.size() - 1,
                             max_digit_flip,
                             std::to_string(api_.window_pp.get_horizontal_flip()).c_str());

        ui_->RotatePushButton->setText(current_rotation);
        ui_->FlipPushButton->setText(current_flip);
    }

    // Deactivate previous maximum (chetor)
    ui_->PSpinBox->setMaximum(INT_MAX);
    ui_->PAccSpinBox->setMaximum(INT_MAX);

    // p accu
    ui_->PAccSpinBox->setValue(api_.transform.get_p_accu_level());
    ui_->PSpinBox->setValue(api_.transform.get_p_index());
    ui_->PAccSpinBox->setEnabled(api_.compute.get_img_type() != ImgType::PhaseIncrease &&
                                 api_.compute.get_img_type() != ImgType::Composite);
    ui_->PAccSpinBox->setVisible(is_data_not_moments);
    ui_->PAccLabel->setVisible(is_data_not_moments);

    // Enforce maximum value for p_index and p_accu_level
    ui_->PSpinBox->setMaximum(api_.transform.get_time_transformation_size() - api_.transform.get_p_accu_level() - 1);
    ui_->PAccSpinBox->setMaximum(api_.transform.get_time_transformation_size() - api_.transform.get_p_index() - 1);
    ui_->PSpinBox->setEnabled(!is_raw && api_.compute.get_img_type() != ImgType::Composite);
    ui_->PSpinBox->setVisible(is_data_not_moments);
    ui_->PLabel->setVisible(is_data_not_moments);

    // q accu
    bool is_q_visible =
        api_.transform.get_time_transformation() == TimeTransformation::SSA_STFT && !is_raw && is_data_not_moments;
    ui_->Q_AccSpinBox->setVisible(is_q_visible);
    ui_->Q_SpinBox->setVisible(is_q_visible);
    ui_->Q_Label->setVisible(is_q_visible);
    ui_->QaccLabel->setVisible(is_q_visible);

    // Deactivate previous maximum (chetor + unused)
    ui_->Q_SpinBox->setMaximum(INT_MAX);
    ui_->Q_AccSpinBox->setMaximum(INT_MAX);

    ui_->Q_AccSpinBox->setValue(api_.transform.get_q_accu_level());
    ui_->Q_SpinBox->setValue(api_.transform.get_q_index());

    // Enforce maximum value for p_index and p_accu_level
    ui_->Q_SpinBox->setMaximum(api_.transform.get_time_transformation_size() - api_.transform.get_q_accu_level() - 1);
    ui_->Q_AccSpinBox->setMaximum(api_.transform.get_time_transformation_size() - api_.transform.get_q_index() - 1);

    // XY accu
    ui_->XAccSpinBox->setValue(api_.transform.get_x_accu_level());
    ui_->YAccSpinBox->setValue(api_.transform.get_y_accu_level());

    int max_width = 0;
    int max_height = 0;
    if (api_.compute.get_input_queue() != nullptr)
    {
        max_width = api_.input.get_input_fd().width - 1;
        max_height = api_.input.get_input_fd().height - 1;
    }
    else
    {
        api_.transform.set_x_cuts(0);
        api_.transform.set_y_cuts(0);
    }

    ui_->XSpinBox->setMaximum(max_width);
    ui_->YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui_->XSpinBox, api_.transform.get_x_cuts());
    QSpinBoxQuietSetValue(ui_->YSpinBox, api_.transform.get_y_cuts());

    // XY accu visibility
    bool xy_visible = api_.view.get_cuts_view_enabled();
    ui_->XSpinBox->setVisible(xy_visible);
    ui_->XLabel->setVisible(xy_visible);
    ui_->XAccSpinBox->setVisible(xy_visible);
    ui_->XAccLabel->setVisible(xy_visible);
    ui_->YSpinBox->setVisible(xy_visible);
    ui_->YLabel->setVisible(xy_visible);
    ui_->YAccSpinBox->setVisible(xy_visible);
    ui_->YAccLabel->setVisible(xy_visible);

    ui_->RenormalizeCheckBox->setChecked(api_.global_pp.get_renorm_enabled());
    ui_->ContrastReticleScaleDoubleSpinBox->setEnabled(api_.contrast.get_contrast_reticle_display_enabled());
    ui_->ContrastReticleScaleDoubleSpinBox->setValue(api_.contrast.get_contrast_reticle_scale());
    ui_->DisplayContrastReticleCheckBox->setChecked(api_.contrast.get_contrast_reticle_display_enabled());
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

void ViewPanel::set_img_type(const QString& value)
{
    const ImgType img_type = static_cast<ImgType>(ui_->ViewModeComboBox->currentIndex());

    bool composite = img_type == ImgType::Composite || api_.compute.get_img_type() == ImgType::Composite;

    if (api_.compute.set_img_type(img_type) == ApiCode::OK)
    {
        // Composite need a refresh of the window since the depth has changed.
        // A better way would be to just update the buffer and texParam of OpenGL
        if (composite)
            gui::refresh_window(parent_->window_max_size);

        parent_->notify();
    }
}

void ViewPanel::set_unwrapping_2d(const bool value)
{
    api_.transform.set_unwrapping_2d(value);
    parent_->notify();
}

void ViewPanel::update_3d_cuts_view(bool checked)
{
    if (api_.view.set_3d_cuts_view(checked))
    {
        gui::set_3d_cuts_view(checked, time_transformation_cuts_window_max_size);
        parent_->notify(); // Make the x and y parameters visible
    }
}

void ViewPanel::set_fft_shift(const bool value)
{
    api_.transform.set_fft_shift_enabled(value);
    parent_->notify();
}

void ViewPanel::set_registration(bool value)
{
    api_.global_pp.set_registration_enabled(value);
    parent_->notify();
}

void ViewPanel::update_lens_view(bool checked)
{
    api_.view.set_lens_view(checked);
    gui::set_lens_view(checked, parent_->auxiliary_window_max_size);
}

void ViewPanel::update_raw_view(bool checked)
{
    api_.view.set_raw_view(checked);
    gui::set_raw_view(checked, parent_->auxiliary_window_max_size);
}

void ViewPanel::set_x_y()
{
    api_.transform.set_x_cuts(ui_->XSpinBox->value());
    api_.transform.set_y_cuts(ui_->YSpinBox->value());
    parent_->notify();
}

void ViewPanel::set_x_accu() { api_.transform.set_x_accu_level(ui_->XAccSpinBox->value()); }

void ViewPanel::set_y_accu() { api_.transform.set_y_accu_level(ui_->YAccSpinBox->value()); }

void ViewPanel::set_p(int value)
{
    api_.transform.set_p_index(value);
    parent_->notify();
}

void ViewPanel::increment_p() { set_p(api_.transform.get_p_index() + 1); }

void ViewPanel::decrement_p() { set_p(api_.transform.get_p_index() - 1); }

void ViewPanel::set_p_accu()
{
    api_.transform.set_p_accu_level(ui_->PAccSpinBox->value());
    parent_->notify();
}

void ViewPanel::set_q(int value)
{
    api_.transform.set_q_index(value);
    parent_->notify();
}

void ViewPanel::set_q_acc()
{
    api_.transform.set_q_accu_level(ui_->Q_AccSpinBox->value());
    parent_->notify();
}

void ViewPanel::rotate_texture()
{
    gui::rotate_texture();
    parent_->notify(); // Update rotate number
}

void ViewPanel::flip_texture()
{
    gui::flip_texture();
    parent_->notify(); // Update flip number
}

void ViewPanel::set_log_enabled(const bool value) { api_.contrast.set_log_enabled(value); }

void ViewPanel::set_accumulation_level(int value) { api_.window_pp.set_accumulation_level(value); }

void ViewPanel::set_contrast_mode(bool value)
{
    api_.contrast.set_contrast_enabled(value);
    parent_->notify();
}

void ViewPanel::set_contrast_auto_refresh(bool value)
{
    api_.contrast.set_contrast_auto_refresh(value);
    parent_->notify();
}

void ViewPanel::enable_contrast_auto_refresh() { set_contrast_auto_refresh(true); }

void ViewPanel::disable_contrast_auto_refresh() { set_contrast_auto_refresh(false); }

void ViewPanel::set_contrast_invert(bool value) { api_.contrast.set_contrast_invert(value); }

void ViewPanel::set_contrast_min(const double value) { api_.contrast.set_contrast_min(value); }

void ViewPanel::set_contrast_max(const double value) { api_.contrast.set_contrast_max(value); }

void ViewPanel::toggle_renormalize(bool value) { api_.global_pp.set_renorm_enabled(value); }

void ViewPanel::display_contrast_reticle(bool value)
{
    api_.contrast.set_contrast_reticle_display_enabled(value);
    gui::set_contrast_reticle_overlay_visible(value);
    parent_->notify();
}

void ViewPanel::contrast_reticle_scale(double value)
{
    api_.contrast.set_contrast_reticle_scale(value);
    gui::set_contrast_reticle_overlay_visible(true);
}

void ViewPanel::display_reticle(bool value)
{
    api_.contrast.set_reticle_display_enabled(value);
    gui::set_reticle_overlay_visible(value);
    parent_->notify();
}

void ViewPanel::reticle_scale(double value)
{
    api_.contrast.set_reticle_scale(value);
    gui::set_reticle_overlay_visible(true);
}

void ViewPanel::update_registration_zone(double value)
{
    api_.global_pp.set_registration_zone(value);

    if (UserInterfaceDescriptor::instance().mainDisplay)
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().enable<gui::Registration>(false, 1000);
}
} // namespace holovibes::gui
