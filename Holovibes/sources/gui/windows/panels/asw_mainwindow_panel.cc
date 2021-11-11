#include "asw_mainwindow_panel.hh"

namespace holovibes::gui
{

ASWMainWindowPanel::ASWMainWindowPanel(double* z_step)
    : AdvancedSettingsWindowPanel("MainWindow")
{
    chart_layout_ = new QVBoxLayout();

    // Widgets creation
    create_z_step_widget(z_step);

    setLayout(chart_layout_);
}

ASWMainWindowPanel::~ASWMainWindowPanel() {}

#pragma region WIDGETS

void ASWMainWindowPanel::create_z_step_widget(double* z_step)
{
    // z step spin box
    z_step_ = new QDoubleRefSpinBoxLayout(nullptr, "z_step", z_step);
    chart_layout_->addItem(z_step_);
}

#pragma endregion

} // namespace holovibes::gui