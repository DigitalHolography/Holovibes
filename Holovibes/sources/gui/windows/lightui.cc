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
    : QDialog(parent)
    , ui_(new Ui::LightUI), main_window_(main_window), export_panel_(main_window->get_export_panel())
{
    ui_->setupUi(this);

    /*
    QHBoxLayout *outputFileSelectionLayout;
    holovibes::gui::Drag_drop_lineedit *OutputFilePathLineEdit;
    QToolButton *OutputFileBrowseToolButton;
    QPushButton *startButton;
    */
    connect(ui_->OutputFileBrowseToolButton, &QPushButton::clicked, this, &LightUI::browse_record_output_file_ui);
    connect(ui_->OutputFilePathLineEdit, &QLineEdit::textChanged, this, &LightUI::update_record_file_path);
    connect(ui_->startButton, &QPushButton::toggled, this, &LightUI::start_stop_recording);
}

LightUI::~LightUI()
{
    delete ui_;
}

void LightUI::browse_record_output_file_ui() {
    LOG_INFO("Browsing record output file");
    export_panel_->browse_record_output_file();
//     QString filepath;

//     // Open file explorer dialog on the fly depending on the record mode
//     // Add the matched extension to the file if none
//     if (api::get_record_mode() == RecordMode::CHART)
//     {
//         filepath = QFileDialog::getSaveFileName(this,
//                                                 tr("Chart output file"),
//                                                 UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
//                                                 tr("Text files (*.txt);;CSV files (*.csv)"));
//     }
//     else if (api::get_record_mode() == RecordMode::RAW)
//     {
//         filepath = QFileDialog::getSaveFileName(this,
//                                                 tr("Record output file"),
//                                                 UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
//                                                 tr("Holo files (*.holo)"));
//     }
//     else if (api::get_record_mode() == RecordMode::HOLOGRAM)
//     {
//         filepath = QFileDialog::getSaveFileName(this,
//                                                 tr("Record output file"),
//                                                 UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
//                                                 tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
//     }
//     else if (api::get_record_mode() == RecordMode::CUTS_XZ || api::get_record_mode() == RecordMode::CUTS_YZ)
//     {
//         filepath = QFileDialog::getSaveFileName(this,
//                                                 tr("Record output file"),
//                                                 UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
//                                                 tr("Mp4 files (*.mp4);; Avi Files (*.avi);;"));
//     }

//     if (filepath.isEmpty())
//         return;

//     // Convert QString to std::string
//     std::string std_filepath = filepath.toStdString();
//     ui_->OutputFilePathLineEdit->setText(filepath);
//     const std::string file_ext = api::browse_record_output_file(std_filepath);
//     // Will pick the item combobox related to file_ext if it exists, else, nothing is done
//     // ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

//     // parent_->notify();
}

void LightUI::update_record_file_path() {
    LOG_INFO("Updating record file path");
    api::set_record_file_path(ui_->OutputFilePathLineEdit->text().toStdString() +
                              ".holo"); // ui_->RecordExtComboBox->currentText().toStdString());
}

void LightUI::start_stop_recording(bool start) {
    char str[20];
    if (start) {
        strcpy(str, "Start recording");
        export_panel_->start_record();
    } else {
        strcpy(str, "Stop recording");
        export_panel_->stop_record();
    }
    LOG_INFO(str);
}

}