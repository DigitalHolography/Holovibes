#include <filesystem>

#include "import_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "input_frame_file_factory.hh"

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
    file_input_directory_ = "C:\\";
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify() { ui_->InputBrowseToolButton->setEnabled(cd_.is_computation_stopped); }

void ImportPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    ui_->actionImportExport->setChecked(!ptree.get<bool>("import_export.hidden", isHidden()));
    file_input_directory_ = ptree.get<std::string>("files.file_input_directory", file_input_directory_);
    ui_->ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));
}

void ImportPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<bool>("import_export.hidden", ui_->ImportExportFrame->isHidden());
    ptree.put<std::string>("files.file_input_directory", file_input_directory_);
}

ImportPanel::ImportType ImportPanel::get_import_type() { return import_type_; }

void ImportPanel::set_import_type(ImportPanel::ImportType type) { import_type_ = type; }

std::string& ImportPanel::get_file_input_directory() { return file_input_directory_; }

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
    filename = QFileDialog::getOpenFileName(this,
                                            tr("import file"),
                                            QString::fromStdString(file_input_directory_),
                                            tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files "
                                               "(*.cine)"));
    LOG_INFO << filename.toStdString();

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

    if (!filename.isEmpty())
    {
        try
        {
            // Will throw if the file format (extension) cannot be handled
            io_files::InputFrameFile* input_file = io_files::InputFrameFileFactory::open(filename.toStdString());

            // Gather data from the newly opened file
            size_t nb_frames = input_file->get_total_nb_frames();
            file_fd_ = input_file->get_frame_descriptor();
            input_file->import_compute_settings(cd_);

            // Don't need the input file anymore
            delete input_file;

            // Update the ui with the gathered data
            ui_->ImportEndIndexSpinBox->setMaximum(nb_frames);
            ui_->ImportEndIndexSpinBox->setValue(nb_frames);

            // We can now launch holovibes over this file
            set_start_stop_buttons(true);
        }
        catch (const io_files::FileException& e)
        {
            // In case of bad format, we triggered the user
            QMessageBox messageBox;
            messageBox.critical(nullptr, "File Error", e.what());
            LOG_ERROR << e.what();

            // Holovibes cannot be launched over this file
            set_start_stop_buttons(false);
        }
    }

    else
        set_start_stop_buttons(false);
}

void ImportPanel::import_stop()
{
    parent_->close_windows();
    ui_->ViewPanel->cancel_time_transformation_cuts();

    parent_->holovibes.stop_all_worker_controller();
    parent_->holovibes.start_information_display(false);

    parent_->close_critical_compute();

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    parent_->camera_none();

    cd_.set_computation_stopped(true);

    parent_->notify();
}

void ImportPanel::import_start()
{
    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    parent_->move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));

    if (!cd_.is_computation_stopped)
        // if computation is running
        import_stop();

    cd_.set_computation_stopped(false);
    // Gather all the useful data from the ui import panel
    init_holovibes_import_mode();

    ui_->ImageModeComboBox->setCurrentIndex(parent_->is_raw_mode() ? 0 : 1);
}

void ImportPanel::init_holovibes_import_mode()
{
    // Get all the useful ui items
    QLineEdit* import_line_edit = ui_->ImportPathLineEdit;
    QSpinBox* fps_spinbox = ui_->ImportInputFpsSpinBox;
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui_->LoadFileInGpuCheckBox;
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    // Set the image rendering ui params
    cd_.set_rendering_params(static_cast<float>(fps_spinbox->value()));

    // Because we are in import mode
    parent_->is_enabled_camera_ = false;

    try
    {
        // Gather data from import panel
        std::string file_path = import_line_edit->text().toStdString();
        unsigned int fps = fps_spinbox->value();
        uint first_frame = start_spinbox->value();
        uint last_frame = end_spinbox->value();
        bool load_file_in_gpu = load_file_gpu_box->isChecked();

        parent_->holovibes.init_input_queue(file_fd_);
        parent_->holovibes.start_file_frame_read(file_path,
                                                 true,
                                                 fps,
                                                 first_frame - 1,
                                                 last_frame - first_frame + 1,
                                                 load_file_in_gpu,
                                                 [=]() {
                                                     parent_->synchronize_thread([&]() {
                                                         if (cd_.is_computation_stopped)
                                                             ui_->InfoPanel->set_visible_file_reader_progress(false);
                                                     });
                                                 });
        ui_->InfoPanel->set_visible_file_reader_progress(true);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        parent_->is_enabled_camera_ = false;
        parent_->mainDisplay.reset(nullptr);
        parent_->holovibes.stop_compute();
        parent_->holovibes.stop_frame_read();
        return;
    }

    parent_->is_enabled_camera_ = true;
    ui_->ImageRenderingPanel->set_image_mode(nullptr);

    // Make camera's settings menu unaccessible
    QAction* settings = ui_->actionSettings;
    settings->setEnabled(false);

    import_type_ = ImportType::File;

    parent_->notify();
}

void ImportPanel::import_start_spinbox_update()
{
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    if (start_spinbox->value() > end_spinbox->value())
        end_spinbox->setValue(start_spinbox->value());
}

void ImportPanel::import_end_spinbox_update()
{
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    if (end_spinbox->value() < start_spinbox->value())
        start_spinbox->setValue(end_spinbox->value());
}
} // namespace holovibes::gui
