#include "lightui.hh"
// #include "export_panel.hh"
#include "MainWindow.hh"
#pragma warning(push, 0)
#include "ui_lightui.h"
#pragma warning(pop)

namespace holovibes::gui
{
LightUI::LightUI(QWidget *parent, MainWindow* main_window)
    : QDialog(parent)
    , ui_(new Ui::LightUI), main_window_(main_window)
{
    ui_->setupUi(this);

    /*
    QHBoxLayout *outputFileSelectionLayout;
    holovibes::gui::Drag_drop_lineedit *OutputFilePathLineEdit;
    QToolButton *OutputFileBrowseToolButton;
    QPushButton *startButton;
    */
    connect(ui_->OutputFileBrowseToolButton, &QPushButton::clicked, this, &LightUI::browse_record_output_file);
    connect(ui_->OutputFilePathLineEdit, &QLineEdit::textChanged, this, &LightUI::update_record_file_path);
    connect(ui_->startButton, &QPushButton::toggled, this, &LightUI::start_stop_recording);
}

LightUI::~LightUI()
{
    delete ui_;
}

void LightUI::browse_record_output_file() {
    LOG_INFO("Browsing record output file");
}

void LightUI::update_record_file_path() {
    LOG_INFO("Updating record file path");
}

void LightUI::start_stop_recording(bool start) {
    char str[20];
    if (start)
        strcpy(str, "Start recording");
    else
        strcpy(str, "Stop recording");
    LOG_INFO(str);
}

void LightUI::set_nb_frames_mode(bool value) {
    char str[50];
    if (value)
        strcpy(str, "Enable number of frame restriction");
    else
        strcpy(str, "Disable number of frame restriction");
    LOG_INFO(str);
}

}