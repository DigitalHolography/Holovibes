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

void AnalysisPanel::on_notify() {}

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

} // namespace holovibes::gui