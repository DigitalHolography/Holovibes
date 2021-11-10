#include "asw_panel_chart.hh"

namespace holovibes::gui
{

ASWPanelChart::ASWPanelChart(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Chart")
{
    chart_layout_ = new QVBoxLayout();

    // File spin box
    auto_scale_point_threshold_ = new QIntSpinBoxLayout(parent, parent_widget, "DisplayRate");
    auto_scale_point_threshold_->setValue(100);
    chart_layout_->addItem(auto_scale_point_threshold_);
    connect(auto_scale_point_threshold_,
            SIGNAL(value_changed()),
            this,
            SLOT(on_change_auto_scale_point_threshold_value()));

    setLayout(chart_layout_);
}

ASWPanelChart::~ASWPanelChart() {}

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelChart::on_change_auto_scale_point_threshold_value()
{
    LOG_INFO << auto_scale_point_threshold_->get_value();
}

#pragma endregion

} // namespace holovibes::gui