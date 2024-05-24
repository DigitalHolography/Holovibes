#include <filesystem>

#include "lightui.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"
#pragma warning(push, 0)
#include "ui_lightui.h"
#pragma warning(pop)

namespace holovibes::gui
{
LightUI::LightUI(QWidget *parent, MainWindow* main_window, ExportPanel* export_panel)
    : QMainWindow(parent)
    , ui_(new Ui::LightUI), main_window_(main_window), export_panel_(main_window->get_export_panel()), visible_(false)
{
    ui_->setupUi(this);

    /*
    QHBoxLayout *outputFileSelectionLayout;
    holovibes::gui::Drag_drop_lineedit *OutputFilePathLineEdit;
    QToolButton *OutputFileBrowseToolButton;
    QPushButton *startButton;
    */
    connect(ui_->OutputFileBrowseToolButton, &QPushButton::clicked, this, &LightUI::browse_record_output_file_ui);
    connect(ui_->startButton, &QPushButton::toggled, this, &LightUI::start_stop_recording);
    connect(ui_->actionConfiguration_UI, &QAction::triggered, this, &LightUI::open_configuration_ui);

    main_window_->set_light_ui(this);
}

LightUI::~LightUI()
{
    api::write_ui_mode(visible_);

    delete ui_;
}

void LightUI::showEvent(QShowEvent *event) 
{
    QMainWindow::showEvent(event);
    visible_ = true;
}

void LightUI::browse_record_output_file_ui() {
    LOG_INFO("Browsing record output file");
    ui_->OutputFilePathLineEdit->setText(export_panel_->browse_record_output_file());
}

void LightUI::start_stop_recording(bool start) {
    char str[20];
    if (start) {
        strcpy(str, "Start recording");
        export_panel_->start_record();
        ui_->startButton->setText("Stop");
        ui_->startButton->setStyleSheet("background-color: rgb(0, 0, 255);");
    } else {
        strcpy(str, "Stop recording");
        export_panel_->stop_record();
        ui_->startButton->setText("Start");
        ui_->startButton->setStyleSheet("background-color: rgb(0, 0, 0);");
    }
    LOG_INFO(str);
}

void LightUI::open_configuration_ui() {
    LOG_INFO("Opening configuration UI");
    main_window_->show();
    this->hide();
    visible_ = false;
}

} // namespace holovibes::gui