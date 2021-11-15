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

void ImportPanel::init()
{
    // Fix for the placeholder color that was black by default because of QT6
    auto p = ui_->ImportPathLineEdit->palette();
    p.setColor(QPalette::PlaceholderText, Qt::darkGray);
    ui_->ImportPathLineEdit->setPalette(p);
}

void ImportPanel::on_notify() { ui_->InputBrowseToolButton->setEnabled(api::get_is_computation_stopped()); }

void ImportPanel::load_gui(const boost::property_tree::ptree& ptree)
{
    bool h = ptree.get<bool>("window.import_export_hidden", ui_->ImportExportFrame->isHidden());
    ui_->actionImportExport->setChecked(!h);
    ui_->ImportExportFrame->setHidden(h);

    ui_->ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));
    ui_->LoadFileInGpuCheckBox->setChecked(ptree.get<bool>("import.from_gpu", false));
}

void ImportPanel::save_gui(boost::property_tree::ptree& ptree)
{
    ptree.put<bool>("window.import_export_hidden", ui_->ImportExportFrame->isHidden());

    ptree.put<uint>("import.fps", ui_->ImportInputFpsSpinBox->value());
    ptree.put<bool>("import.from_gpu", ui_->LoadFileInGpuCheckBox->isChecked());
}

ImportType ImportPanel::get_import_type() { return UserInterfaceDescriptor::instance().import_type_; }

void ImportPanel::set_import_type(ImportType type) { UserInterfaceDescriptor::instance().import_type_ = type; }

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
        LOG_ERROR << e.what();

        // Holovibes cannot be launched over this file
        set_start_stop_buttons(false);
        return;
    }

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();
        // Gather data from the newly opened file
        size_t nb_frames = input_file->get_total_nb_frames();
        UserInterfaceDescriptor::instance().file_fd_ = input_file->get_frame_descriptor();
        input_file->import_compute_settings(api::get_cd());

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
    api::close_windows();
    // cancel_time_transformation_cuts();

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

    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    parent_->move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));

    // Get all the useful ui items
    QLineEdit* import_line_edit = ui_->ImportPathLineEdit;
    QSpinBox* fps_spinbox = ui_->ImportInputFpsSpinBox;
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui_->LoadFileInGpuCheckBox;
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    std::string file_path = import_line_edit->text().toStdString();

    bool res_import_start = api::import_start(file_path,
                                              fps_spinbox->value(),
                                              start_spinbox->value(),
                                              load_file_gpu_box->isChecked(),
                                              end_spinbox->value());

    if (res_import_start)
    {
        ui_->FileReaderProgressBar->show();
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        ui_->ImageRenderingPanel->set_image_mode(nullptr);

        // Make camera's settings menu unaccessible
        QAction* settings = ui_->actionSettings;
        settings->setEnabled(false);

        UserInterfaceDescriptor::instance().import_type_ = ImportType::File;

        parent_->notify();
    }
    else
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }

    ui_->ImageModeComboBox->setCurrentIndex(api::is_raw_mode() ? 0 : 1);
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
