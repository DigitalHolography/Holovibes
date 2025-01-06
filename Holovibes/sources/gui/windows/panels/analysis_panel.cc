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

    // Show arteries / veins
    ui_->ArteryCheckBox->setChecked(API.analysis.get_artery_mask_enabled());
    ui_->VeinCheckBox->setChecked(API.analysis.get_vein_mask_enabled());
    ui_->ChoroidCheckBox->setChecked(API.analysis.get_choroid_mask_enabled());

    // Time window
    ui_->TimeWindowSpinBox->setValue(API.analysis.get_time_window());

    // Vesselness sigma
    ui_->VesselnessSigmaDoubleSpinBox->setValue(API.analysis.get_vesselness_sigma());
    ui_->VesselnessSigmaSlider->setValue((int)std::round(API.analysis.get_vesselness_sigma() * 100));

    // Min mask Area
    ui_->MinMaskAreaSpinBox->setValue(API.analysis.get_min_mask_area());
    ui_->MinMaskAreaSlider->setValue(API.analysis.get_min_mask_area());

    // Diaphragm
    ui_->DiaphragmFactorDoubleSpinBox->setValue(API.analysis.get_diaphragm_factor());
    ui_->DiaphragmFactorSlider->setValue(API.analysis.get_diaphragm_factor() * 1000);
    ui_->DiaphragmPreviewCheckBox->setChecked(API.analysis.get_diaphragm_preview_enabled());

    // Barycenter
    ui_->BarycenterFactorDoubleSpinBox->setValue(API.analysis.get_barycenter_factor());
    ui_->BarycenterFactorSlider->setValue(API.analysis.get_barycenter_factor() * 1000);
    ui_->BarycenterPreviewCheckBox->setChecked(API.analysis.get_barycenter_preview_enabled());
}

void AnalysisPanel::set_artery_mask(bool enabled)
{
    API.analysis.set_artery_mask_enabled(enabled);
    ui_->ArteryCheckBox->setChecked(enabled);
}

void AnalysisPanel::set_vein_mask(bool enabled)
{
    API.analysis.set_vein_mask_enabled(enabled);
    ui_->VeinCheckBox->setChecked(enabled);
}

void AnalysisPanel::set_choroid_mask(bool enabled)
{
    API.analysis.set_choroid_mask_enabled(enabled);
    ui_->ChoroidCheckBox->setChecked(enabled);
}

void AnalysisPanel::update_time_window()
{
    QSpinBox* time_window_spinbox = ui_->TimeWindowSpinBox;

    API.analysis.set_time_window(time_window_spinbox->value());

    time_window_spinbox->setValue(API.analysis.get_time_window());
    ui_->TimeWindowSpinBox->setValue(API.analysis.get_time_window());
}

void AnalysisPanel::update_vesselness_sigma(double value)
{
    API.analysis.set_vesselness_sigma(value);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->VesselnessSigmaSlider);
    ui_->VesselnessSigmaSlider->setValue(value * 100);
}

void AnalysisPanel::update_vesselness_sigma_slider(int value)
{
    double new_value = value / 100.0f;
    API.analysis.set_vesselness_sigma(new_value);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->VesselnessSigmaDoubleSpinBox);
    ui_->VesselnessSigmaDoubleSpinBox->setValue(new_value);
}

void AnalysisPanel::update_min_mask_area(int value)
{
    API.analysis.set_min_mask_area(value);
    ui_->MinMaskAreaSlider->setValue(value);
}

void AnalysisPanel::update_min_mask_area_slider(int value)
{
    API.analysis.set_min_mask_area(value);
    ui_->MinMaskAreaSpinBox->setValue(value);
}

void AnalysisPanel::update_diaphragm_factor(double value)
{
    API.analysis.set_diaphragm_factor(value);
    ui_->DiaphragmFactorSlider->setValue(value * 1000);
}

void AnalysisPanel::update_diaphragm_factor_slider(int value)
{
    double real_value = (double)value / 1000;
    API.analysis.set_diaphragm_factor(real_value);
    ui_->DiaphragmFactorDoubleSpinBox->setValue(real_value);
}

void AnalysisPanel::update_diaphragm_preview(bool enabled) { API.analysis.set_diaphragm_preview_enabled(enabled); }

void AnalysisPanel::update_barycenter_factor(double value)
{
    API.analysis.set_barycenter_factor(value);
    ui_->BarycenterFactorSlider->setValue(value * 1000);
}

void AnalysisPanel::update_barycenter_factor_slider(int value)
{
    double real_value = (double)value / 1000;
    API.analysis.set_barycenter_factor(real_value);
    ui_->BarycenterFactorDoubleSpinBox->setValue(real_value);
}

void AnalysisPanel::update_barycenter_preview(bool enabled) { API.analysis.set_barycenter_preview_enabled(enabled); }

} // namespace holovibes::gui
