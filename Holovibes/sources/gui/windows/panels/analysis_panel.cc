/*! \file
 *
 */

#include <limits>

#include "analysis_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"

#include "view_struct.hh"

#include "API.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
AnalysisPanel::AnalysisPanel(QWidget* parent)
    : Panel(parent)
{
}

AnalysisPanel::~AnalysisPanel() {}

void AnalysisPanel::init() {}

void AnalysisPanel::on_notify()
{
    // Updating UI values with current backend settings

    // Otsu
    ui_->OtsuCheckBox->setChecked(api::get_otsu_enabled());

    ui_->OtsuKindComboBox->setCurrentIndex(static_cast<int>(api::get_otsu_kind()));
    bool is_adaptive = api::get_otsu_kind() == OtsuKind::Adaptive;
    ui_->OtsuWindowSizeSpinBox->setVisible(is_adaptive);
    ui_->OtsuWindowSizeLabel->setVisible(is_adaptive);
    ui_->OtsuLocalThresholdSpinBox->setVisible(is_adaptive);
    ui_->OtsuLocalThresholdLabel->setVisible(is_adaptive);

    ui_->OtsuWindowSizeSpinBox->setValue(api::get_otsu_window_size());
    ui_->OtsuWindowSizeSpinBox->setEnabled(api::get_otsu_enabled());
    ui_->OtsuWindowSizeLabel->setEnabled(api::get_otsu_enabled());
    ui_->OtsuLocalThresholdSpinBox->setValue(api::get_otsu_local_threshold());
    ui_->OtsuLocalThresholdSpinBox->setEnabled(api::get_otsu_enabled());
    ui_->OtsuLocalThresholdLabel->setEnabled(api::get_otsu_enabled());
}

void AnalysisPanel::update_time_window()
{
    QSpinBox* time_window_spinbox = ui_->TimeWindowSpinBox;

    api::set_time_window(time_window_spinbox->value());

    time_window_spinbox->setValue(api::get_time_window());
    ui_->TimeWindowSpinBox->setValue(api::get_time_window());
}

void AnalysisPanel::update_vesselness_sigma(double value)
{
    api::set_vesselness_sigma(value);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->VesselnessSigmaSlider);
    ui_->VesselnessSigmaSlider->setValue(value * 100);
}

void AnalysisPanel::update_vesselness_sigma_slider(int value)
{
    double new_value = value / 100.0f;
    api::set_vesselness_sigma(new_value);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->VesselnessSigmaDoubleSpinBox);
    ui_->VesselnessSigmaDoubleSpinBox->setValue(new_value);
}

void AnalysisPanel::set_otsu_kind(int index)
{
    api::set_otsu_kind(static_cast<OtsuKind>(index));
    parent_->notify();
}

void AnalysisPanel::set_otsu_window_size(int value) { api::set_otsu_window_size(value); }

void AnalysisPanel::set_otsu_local_threshold(double value) { api::set_otsu_local_threshold((float)value); }

} // namespace holovibes::gui
