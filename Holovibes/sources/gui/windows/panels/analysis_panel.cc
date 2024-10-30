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

    // Bw area filter
    // TODO

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

void AnalysisPanel::set_otsu_kind(int index)
{
    api::set_otsu_kind(static_cast<OtsuKind>(index));
    parent_->notify();
}

void AnalysisPanel::set_otsu_window_size(int value) { api::set_otsu_window_size(value); }

void AnalysisPanel::set_otsu_local_threshold(double value) { api::set_otsu_local_threshold((float)value); }

void AnalysisPanel::set_bw_area_filter(bool enabled) { api::set_bwareafilt_enabled(enabled); }

void AnalysisPanel::set_bw_area_filter_value(int value) { api::set_bwareafilt_n(value); }
} // namespace holovibes::gui
