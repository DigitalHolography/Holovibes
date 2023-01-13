/*! \file
 *
 */

#include <filesystem>

#include "import_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "input_frame_file_factory.hh"
#include "API.hh"

#include "user_interface.hh"
#include "information_worker.hh"
#include <QtGui/QPalette>

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().import_panel = this;

    // ::holovibes::worker::InformationWorker::is_input_queue_ok_ = [&](bool value)
    // {
    //     UserInterface::instance().main_window->synchronize_thread(
    //         [&]()
    //         {
    //             QPalette palette = ui_->FileReaderProgressBar->palette();
    //             if (value)
    //                 palette.setColor(QPalette::Highlight, Qt::blue);
    //             else
    //                 palette.setColor(QPalette::Highlight, Qt::red);
    //             ui_->FileReaderProgressBar->setPalette(palette);
    //         },
    //         true);
    // };
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify()
{
    ui_->InputBrowseToolButton->setEnabled(api::get_import_type() == ImportTypeEnum::None);

    ui_->ImportStartPushButton->setEnabled(!api::get_import_file_path().empty() &&
                                           api::get_import_type() == ImportTypeEnum::None);

    ui_->ImportStopPushButton->setEnabled(!api::get_import_file_path().empty() &&
                                          api::get_import_type() != ImportTypeEnum::None);

    ui_->ImportStartIndexSpinBox->setMinimum(1);
    ui_->ImportStartIndexSpinBox->setValue(api::get_start_frame());

    ui_->ImportEndIndexSpinBox->setMaximum(api::get_file_number_of_frames());
    ui_->ImportEndIndexSpinBox->setMinimum(1);
    ui_->ImportEndIndexSpinBox->setValue(api::get_end_frame());

    if (api::get_import_type() != ImportTypeEnum::None)
        ui_->FileReaderProgressBar->show();
    else
        ui_->FileReaderProgressBar->hide();
}

void ImportPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, ui_->ImportExportFrame->isHidden(), "panels", "import export hidden");
    ui_->actionImportExport->setChecked(!h);
    ui_->ImportExportFrame->setHidden(h);

    ui_->ImportInputFpsSpinBox->setValue(json_get_or_default(j_us, 60, "import", "fps"));
    ui_->LoadFileInGpuCheckBox->setChecked(json_get_or_default(j_us, false, "import", "from gpu"));
}

void ImportPanel::save_gui(json& j_us)
{
    j_us["panels"]["import export hidden"] = ui_->ImportExportFrame->isHidden();

    j_us["import"]["fps"] = ui_->ImportInputFpsSpinBox->value();
    j_us["import"]["from gpu"] = ui_->LoadFileInGpuCheckBox->isChecked();
}

std::string& ImportPanel::get_file_input_directory() { return UserInterface::instance().file_input_directory_; }

void ImportPanel::import_browse_file()
{
    QString filename = "";

    // Open the file explorer to let the user pick his file
    // and store the chosen file in filename
    filename = QFileDialog::getOpenFileName(this,
                                            tr("import file"),
                                            QString::fromStdString(UserInterface::instance().file_input_directory_),
                                            tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files "
                                               "(*.cine)"));

    // Start importing the chosen
    import_file(filename);
}

void ImportPanel::import_file(const QString& filename)
{
    ui_->ImportPathLineEdit->clear();
    ui_->ImportPathLineEdit->insert(filename);

    api::detail::set_value<ImportFilePath>(filename.toStdString());
}

// clang-format off
void ImportPanel::import_stop()
{
    api::set_import_type(ImportTypeEnum::None);
}
// clang-format on

// TODO: review function, we cannot edit UserInterface here (instead of API)
void ImportPanel::import_start()
{
    parent_->shift_screen();
    ui_->FileReaderProgressBar->show();

    api::set_load_in_gpu(ui_->LoadFileInGpuCheckBox->isChecked());
    api::set_import_type(ImportTypeEnum::File);
}

void ImportPanel::import_start_spinbox_update() { api::set_start_frame(ui_->ImportStartIndexSpinBox->value()); }

void ImportPanel::import_end_spinbox_update() { api::set_end_frame(ui_->ImportEndIndexSpinBox->value()); }

void ImportPanel::on_input_fps_change(int value) { api::set_input_fps(value); }

} // namespace holovibes::gui
