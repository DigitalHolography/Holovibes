/*! \file
 *
 */

#include "asw_mainwindow_panel.hh"

namespace holovibes::gui
{

ASWMainWindowPanel::ASWMainWindowPanel(MainWindow* parent)
    : AdvancedSettingsWindowPanel("MainWindow")
    , parent_(parent)
{
    UserInterface::instance().asw_main_window_panel = this;
    QVBoxLayout* mainwindow_layout = new QVBoxLayout();

    // Widgets creation
    create_z_step_widget(mainwindow_layout);
    create_record_frame_step_widget(mainwindow_layout);

    setLayout(mainwindow_layout);
    setMaximumSize(210, 2000);
}

ASWMainWindowPanel::~ASWMainWindowPanel() {}

void ASWMainWindowPanel::set_ui_values()
{
    parent_->ui_->ImageRenderingPanel->set_z_step(z_step_->get_value());
    parent_->ui_->ExportPanel->set_record_frame_step(record_frame_step_->get_value());
}

void ASWMainWindowPanel::set_current_values()
{
    z_step_->set_value(parent_->ui_->ImageRenderingPanel->get_z_step());
    record_frame_step_->set_value(parent_->ui_->ExportPanel->get_record_frame_step());
}

#pragma region WIDGETS

void ASWMainWindowPanel::create_z_step_widget(QVBoxLayout* layout)
{
    // z step spin box
    z_step_ = new QDoubleSpinBoxLayout(nullptr, "z step");
    z_step_->set_decimals(3)->set_single_step(0.005)->set_label_min_size(100, 0);
    layout->addItem(z_step_);
}

void ASWMainWindowPanel::create_record_frame_step_widget(QVBoxLayout* layout)
{
    // z step spin box
    record_frame_step_ = new QIntSpinBoxLayout(nullptr, "Record frame step");
    record_frame_step_->set_single_step(64)->set_label_min_size(100, 0);
    layout->addItem(record_frame_step_);
}

#pragma endregion

} // namespace holovibes::gui
