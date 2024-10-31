/*! \file
 *
 */

#include <filesystem>

#include "import_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "input_frame_file_factory.hh"
#include "API.hh"
#include <spdlog/spdlog.h>

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify() { ui_->InputBrowseToolButton->setEnabled(api::get_is_computation_stopped()); }

void ImportPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, ui_->ImportExportFrame->isHidden(), "panels", "import export hidden");
    ui_->actionImportExport->setChecked(!h);
    ui_->ImportExportFrame->setHidden(h);

    ui_->ImportInputFpsSpinBox->setValue(json_get_or_default(j_us, 10000, "import", "fps"));
    update_fps(); // Required as it is called `OnEditedFinished` only.

    ui_->LoadFileInGpuCheckBox->setChecked(json_get_or_default(j_us, false, "import", "from gpu"));
}

void ImportPanel::save_gui(json& j_us)
{
    j_us["panels"]["import export hidden"] = ui_->ImportExportFrame->isHidden();

    j_us["import"]["fps"] = ui_->ImportInputFpsSpinBox->value();
    j_us["import"]["from gpu"] = ui_->LoadFileInGpuCheckBox->isChecked();
}

std::string& ImportPanel::get_file_input_directory()
{
    return UserInterfaceDescriptor::instance().file_input_directory_;
}

void ImportPanel::set_start_stop_buttons(bool value)
{
    ui_->ImportStartPushButton->setEnabled(value);
    ui_->ImportStopPushButton->setEnabled(value);
}

void ImportPanel::import_browse_file()
{
    QString filename = "";

    // Open the file explorer to let the user pick his file
    // and store the chosen file in filename
    filename =
        QFileDialog::getOpenFileName(this,
                                     tr("import file"),
                                     QString::fromStdString(UserInterfaceDescriptor::instance().file_input_directory_),
                                     tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files "
                                        "(*.cine)"));

    // Start importing the chosen
    import_file(filename);
}

void ImportPanel::import_file(const QString& filename)
{
    // Get the widget (output bar) from the ui linked to the file explorer
    QLineEdit* import_line_edit = ui_->ImportPathLineEdit;

    // Insert the newly getted path in it
    import_line_edit->clear();
    import_line_edit->insert(filename);

    // Start importing the chosen
    std::optional<io_files::InputFrameFile*> input_file_opt;
    try
    {
        input_file_opt = api::import_file(filename.toStdString());
    }
    catch (const io_files::FileException& e)
    {
        // In case of bad format, we triggered the user
        QMessageBox messageBox;
        messageBox.critical(nullptr, "File Error", e.what());
        LOG_ERROR("Catch {}", e.what());
        // Holovibes cannot be launched over this file
        set_start_stop_buttons(false);
        return;
    }

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();

        // Get the buffer size that will be used to allocate the buffer for reading the file instead of the one from the
        // record
        auto input_buffer_size = api::get_input_buffer_size();
        auto record_buffer_size = api::get_record_buffer_size();

        // Import Compute Settings there before init_pipe to
        // Allocate correctly buffer
        try
        {
            input_file->import_compute_settings();
            input_file->import_info();
        }
        catch (const std::exception& e)
        {
            QMessageBox messageBox;
            messageBox.critical(nullptr, "File Error", e.what());
            LOG_ERROR("Catch {}", e.what());
            LOG_INFO("Compute settings incorrect or file not found. Initialization with default values.");
            api::save_compute_settings(holovibes::settings::compute_settings_filepath);
        }

        // update the buffer size with the old values to avoid surcharging the gpu memory in case of big buffers used
        // when the file was recorded
        api::set_input_buffer_size(input_buffer_size);
        api::set_record_buffer_size(record_buffer_size);

        parent_->notify();

        // Gather data from the newly opened file
        size_t nb_frames = input_file->get_total_nb_frames();
        UserInterfaceDescriptor::instance().file_fd_ = input_file->get_frame_descriptor();

        // Don't need the input file anymore
        delete input_file;

        // Update the ui with the gathered data
        // The start index cannot exceed the end index
        ui_->ImportStartIndexSpinBox->setMaximum(static_cast<int>(nb_frames));
        ui_->ImportEndIndexSpinBox->setMaximum(static_cast<int>(nb_frames));
        ui_->ImportEndIndexSpinBox->setValue(static_cast<int>(nb_frames));

        // We can now launch holovibes over this file
        set_start_stop_buttons(true);
    }
    else
        set_start_stop_buttons(false);
}

void ImportPanel::import_stop()
{
    api::import_stop();

    parent_->synchronize_thread([&]() { ui_->FileReaderProgressBar->hide(); });
    parent_->notify();
}

// TODO: review function, we cannot edit UserInterfaceDescriptor here (instead of API)
void ImportPanel::import_start()
{
    if (api::import_start())
    {
        ui_->FileReaderProgressBar->show();

        // Make camera's settings menu unaccessible
        QAction* settings = ui_->actionSettings;
        settings->setEnabled(false);

        // This notify is required.
        // This sets GUI values and avoid having callbacks destroy and recreate the window and pipe.
        // This prevents a double pipe initialization which is the source of many crashed (for example,
        // going from Raw to Processed using reload_compute_settings or starting a .holo in Hologram mode).
        // Ideally, every value should be set without callbacks before the window is created, which would avoid such
        // problems.
        // This is for now absolutely terrible, but it's a necessary evil until notify is reworked.
        // Something in the notify cancels the convolution. An issue is opened about this problem.
        parent_->notify();

        // Because the previous notify MIGHT create an holo window, we have to create it if it has not been done.
        if (api::get_main_display() == nullptr)
            parent_->ui_->ImageRenderingPanel->set_image_mode(static_cast<int>(api::get_compute_mode()));

        // The reticle overlay needs to be created as soon as the pipe is created, but there isn't many places where
        // this can easily be done while imapcting only the GUI, so it's done here as a dirty fix
        api::display_reticle(api::get_reticle_display_enabled());
    }
    else
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
}

void ImportPanel::update_fps() { api::set_input_fps(ui_->ImportInputFpsSpinBox->value()); }

void ImportPanel::update_import_file_path() { api::set_input_file_path(ui_->ImportPathLineEdit->text().toStdString()); }

void ImportPanel::update_load_file_in_gpu() { api::set_load_file_in_gpu(ui_->LoadFileInGpuCheckBox->isChecked()); }

void ImportPanel::update_input_file_start_index()
{
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;

    api::set_input_file_start_index(start_spinbox->value() - 1);

    start_spinbox->setValue(static_cast<int>(api::get_input_file_start_index()) + 1);
    ui_->ImportEndIndexSpinBox->setValue(static_cast<int>(api::get_input_file_end_index()));
}

void ImportPanel::update_input_file_end_index()
{
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    api::set_input_file_end_index(end_spinbox->value());

    end_spinbox->setValue(static_cast<int>(api::get_input_file_end_index()));
    ui_->ImportStartIndexSpinBox->setValue(static_cast<int>(api::get_input_file_start_index()) + 1);
}

} // namespace holovibes::gui
