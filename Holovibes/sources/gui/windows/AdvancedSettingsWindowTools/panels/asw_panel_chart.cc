#include "asw_panel_chart.hh"

namespace holovibes::gui
{

// TODO: change by API getter call
#define DEFAULT_AUTO_SCALE_POINT_THRESHOLD_VALUE 32

ASWPanelChart::ASWPanelChart(QMainWindow* parent, QWidget* parent_widget)
    : AdvancedSettingsWindowPanel(parent, parent_widget, "Chart")
{
    chart_layout_ = new QVBoxLayout();

    // Widgets creation
    create_auto_scale_point_threshold_widget();

    setLayout(chart_layout_);
}

ASWPanelChart::~ASWPanelChart() {}

#pragma region WIDGETS

void ASWPanelChart::create_auto_scale_point_threshold_widget()
{
    // Auto scale pint threshold spin box
    auto_scale_point_threshold_ = new QIntSpinBoxLayout(parent_, parent_widget_, "auto_scale_point_threshold");
    auto_scale_point_threshold_->setValue(DEFAULT_AUTO_SCALE_POINT_THRESHOLD_VALUE);
    chart_layout_->addItem(auto_scale_point_threshold_);
    connect(auto_scale_point_threshold_,
            SIGNAL(value_changed()),
            this,
            SLOT(on_change_auto_scale_point_threshold_value()));
}

#pragma endregion

#pragma region SLOTS
// TODO: region to implement with API
void ASWPanelChart::on_change_auto_scale_point_threshold_value()
{
    LOG_INFO << auto_scale_point_threshold_->get_value();
}

#pragma endregion

} // namespace holovibes::gui