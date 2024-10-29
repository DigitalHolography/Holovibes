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
    ui_->OtsuWindowSizeSpinBox->setValue(api::get_otsu_window_size());
    ui_->OtsuLocalThresholdSpinBox->setValue(api::get_otsu_local_threshold());
}

void AnalysisPanel::set_otsu_window_size() { api::set_otsu_window_size(ui_->OtsuWindowSizeSpinBox->value()); }

void AnalysisPanel::set_otsu_local_threshold()
{
    api::set_otsu_local_threshold(ui_->OtsuLocalThresholdSpinBox->value());
}

} // namespace holovibes::gui
