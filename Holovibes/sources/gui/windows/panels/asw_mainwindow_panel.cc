#include "asw_mainwindow_panel.hh"

namespace holovibes::gui
{

ASWMainWindowPanel::ASWMainWindowPanel(ImageRenderingPanel* parent)
    : parent_(parent)
    , AdvancedSettingsWindowPanel("MainWindow")
{
    QVBoxLayout* mainwindow_layout = new QVBoxLayout();

    // Widgets creation
    create_z_step_widget(mainwindow_layout);

    setLayout(mainwindow_layout);
}

ASWMainWindowPanel::~ASWMainWindowPanel() {}

void ASWMainWindowPanel::set_ui_values() { parent_->set_z_step(z_step_->get_value()); }

void ASWMainWindowPanel::set_current_values() { z_step_->set_value(parent_->get_z_step()); }

#pragma region WIDGETS

void ASWMainWindowPanel::create_z_step_widget(QVBoxLayout* layout)
{
    // z step spin box
    z_step_ = new QDoubleSpinBoxLayout(nullptr, "z_step");
    z_step_->set_value(parent_->z_step_)
    layout->addItem(z_step_);
}

#pragma endregion

} // namespace holovibes::gui