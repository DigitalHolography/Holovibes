#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
class ASWPanelChart : public AdvancedSettingsWindowPanel
{
  public:
    ASWPanelChart(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelChart();

  private:
    QVBoxLayout* chart_layout_;
    QIntSpinBoxLayout* auto_scale_point_threshold_;
};
} // namespace holovibes::gui