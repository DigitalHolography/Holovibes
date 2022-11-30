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

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().import_panel = this;
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify()
{
    ui_->InputBrowseToolButton->setEnabled(api::get_import_type() == ImportTypeEnum::None);
    ui_->ImportStartPushButton->setEnabled(!api::get_import_file_path().empty());
    ui_->ImportStopPushButton->setEnabled(!api::get_import_file_path().empty());
    ui_->ImportEndIndexSpinBox->setMaximum(api::get_max_end_frame());
    ui_->ImportEndIndexSpinBox->setValue(api::get_max_end_frame());
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
    // Insert the newly getted path in it
    ui_->ImportPathLineEdit->clear();
    ui_->ImportPathLineEdit->insert(filename);

    api::detail::set_value<ImportFilePath>(filename.toStdString());

    // FIXME API - FIXME ERROR RECOVERY
    // Start importing the chosen
    //    try
    //    {
    //    }
    // catch (const io_files::FileException& e)
    //{
    //    // In case of bad format, we triggered the user
    //    QMessageBox messageBox;
    //    messageBox.critical(nullptr, "File Error", e.what());
    //    LOG_ERROR("Catch {}", e.what());
    //    // Holovibes cannot be launched over this file
    //    set_start_stop_buttons(false);
    //    return;
    //}
    //

    // We can now launch holovibes over this file
}

void ImportPanel::import_stop()
{
    api::set_import_type(ImportTypeEnum::None);

    parent_->synchronize_thread([&]() { ui_->FileReaderProgressBar->hide(); });
}

// TODO: review function, we cannot edit UserInterface here (instead of API)
void ImportPanel::import_start()
{
    parent_->shift_screen();
    ui_->FileReaderProgressBar->show();

    // Ensure that all vars are well sync
    api::set_input_fps(ui_->ImportInputFpsSpinBox->value());
    api::set_load_in_gpu(ui_->LoadFileInGpuCheckBox->isChecked());
    api::set_start_frame(ui_->ImportStartIndexSpinBox->value());
    api::set_end_frame(ui_->ImportEndIndexSpinBox->value());
    api::set_import_file_path(ui_->ImportPathLineEdit->text().toStdString());

    api::set_import_type(ImportTypeEnum::File);
}

void ImportPanel::import_start_spinbox_update() { api::set_start_frame(ui_->ImportStartIndexSpinBox->value()); }

void ImportPanel::import_end_spinbox_update() { api::set_end_frame(ui_->ImportEndIndexSpinBox->value()); }

void ImportPanel::on_input_fps_change(int value) { api::set_input_fps(value); }

} // namespace holovibes::gui
