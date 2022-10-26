/*! \file
 *
 */

#include <filesystem>

#include "import_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "input_frame_file_factory.hh"
#include "API.hh"

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

    ui_->ImportInputFpsSpinBox->setValue(json_get_or_default(j_us, 60, "import", "fps"));
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
        LOG_ERROR(main, "Catch {}", e.what());
        // Holovibes cannot be launched over this file
        set_start_stop_buttons(false);
        return;
    }

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();

        // Import Compute Settings there before init_pipe to
        // Allocate correctly buffer
        input_file->import_compute_settings();

        parent_->notify();

        // Gather data from the newly opened file
        size_t nb_frames = input_file->get_total_nb_frames();
        UserInterfaceDescriptor::instance().file_fd_ = input_file->get_frame_descriptor();

        // Don't need the input file anymore
        delete input_file;

        // Update the ui with the gathered data
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
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    api::import_stop();

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    parent_->camera_none();

    parent_->synchronize_thread([&]() { ui_->FileReaderProgressBar->hide(); });
    parent_->notify();
}

// TODO: review function, we cannot edit UserInterfaceDescriptor here (instead of API)
void ImportPanel::import_start()
{
    // Check if computation is currently running
    if (!api::get_is_computation_stopped())
        import_stop();

    parent_->shift_screen();

    // Get all the useful ui items
    QLineEdit* import_line_edit = ui_->ImportPathLineEdit;

    // Now stored in GSH
    // QSpinBox* fps_spinbox = ui_->ImportInputFpsSpinBox;

    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui_->LoadFileInGpuCheckBox;
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    std::string file_path = import_line_edit->text().toStdString();

    bool res_import_start = api::import_start(file_path,
                                              api::get_input_fps(),
                                              start_spinbox->value(),
                                              load_file_gpu_box->isChecked(),
                                              end_spinbox->value());

    if (res_import_start)
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
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }
}

void ImportPanel::import_start_spinbox_update()
{
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;

    api::set_start_frame(start_spinbox->value());

    start_spinbox->setValue(api::get_start_frame());

    if (api::get_start_frame() > api::get_end_frame())
    {
        QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;
        end_spinbox->setValue(api::get_start_frame());
        import_end_spinbox_update();
    }
}

void ImportPanel::import_end_spinbox_update()
{
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    api::set_end_frame(end_spinbox->value());
    end_spinbox->setValue(api::get_end_frame());

    if (api::get_end_frame() < api::get_start_frame())
    {
        QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
        start_spinbox->setValue(api::get_end_frame());
        import_start_spinbox_update();
    }
}

void ImportPanel::on_input_fps_change(int value) { api::set_input_fps(value); }

} // namespace holovibes::gui
