#include "asw_panel_chart.hh"

namespace holovibes::gui
{

ASWPanelChart::ASWPanelChart(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Chart")
{
    chart_layout_ = new QVBoxLayout(parent);

    // File spin box
    auto_scale_point_threshold_ = new QIntSpinBoxLayout(parent, parent_widget, "DisplayRate");
    auto_scale_point_threshold_->setValue(100);
    chart_layout_->addItem(auto_scale_point_threshold_);

    setLayout(chart_layout_);
}

ASWPanelChart::~ASWPanelChart() {}

} // namespace holovibes::gui