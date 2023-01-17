/*! \file
 *
 */

#include "asw_mainwindow_panel.hh"
#include "user_interface.hh"

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

void ASWMainWindowPanel::set_ui_values() {}

void ASWMainWindowPanel::set_current_values() {}

#pragma region WIDGETS

void ASWMainWindowPanel::create_z_step_widget(QVBoxLayout* layout)
{
    // z step spin box
    z_distance_step_ = new QDoubleSpinBoxLayout(nullptr, "z step");
    z_distance_step_->set_decimals(3)->set_single_step(0.005)->set_label_min_size(100, 0);
    layout->addItem(z_distance_step_);
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
