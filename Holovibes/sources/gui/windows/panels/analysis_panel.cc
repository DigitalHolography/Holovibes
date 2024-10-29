/*! \file
 *
 */

#include <filesystem>

#include "analysis_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
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

    ui_->OtsuWindowSizeSpinBox->setValue(api::get_otsu_window_size());
    ui_->OtsuWindowSizeSpinBox->setEnabled(api::get_otsu_enabled());
    ui_->OtsuWindowSizeLabel->setEnabled(api::get_otsu_enabled());
    ui_->OtsuLocalThresholdSpinBox->setValue(api::get_otsu_local_threshold());
    ui_->OtsuLocalThresholdSpinBox->setEnabled(api::get_otsu_enabled());
    ui_->OtsuLocalThresholdLabel->setEnabled(api::get_otsu_enabled());
}

void AnalysisPanel::set_otsu_window_size(int value) { api::set_otsu_window_size(value); }

void AnalysisPanel::set_otsu_local_threshold(double value)
{
    LOG_INFO("Local threshold value: {}", (float)value);
    api::set_otsu_local_threshold((float)value);
}

} // namespace holovibes::gui
