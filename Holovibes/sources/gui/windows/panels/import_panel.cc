/*! \file
 *
 */

#include <filesystem>

#include "import_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "input_frame_file_factory.hh"
#include "API.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"
#include <spdlog/spdlog.h>

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify()
{
    ui_->ImportStartIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_start_index()));
    ui_->ImportEndIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_end_index()));
    const char step = api_.input.get_data_type() == RecordedDataType::MOMENTS ? 3 : 1;
    ui_->ImportStartIndexSpinBox->setSingleStep(step);
    ui_->ImportEndIndexSpinBox->setSingleStep(step);

    const bool no_comp = api_.compute.get_is_computation_stopped();
    ui_->InputBrowseToolButton->setEnabled(no_comp);
    ui_->FileReaderProgressBar->setVisible(!no_comp && api_.input.get_import_type() == ImportType::File);
}

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
    QString filename =
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
    std::optional<io_files::InputFrameFile*> input_file_opt = api_.input.import_file(filename.toStdString());

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();

        parent_->notify();

        // Gather data from the newly opened file
        int nb_frames = static_cast<int>(input_file->get_total_nb_frames());

        // Don't need the input file anymore
        delete input_file;

        // Update the ui with the gathered data
        // The start index cannot exceed the end index
        ui_->ImportStartIndexSpinBox->setMaximum(nb_frames);
        ui_->ImportEndIndexSpinBox->setMaximum(nb_frames);

        // Changing the settings is straight-up better than changing the UI
        // This whole logic will need to go in the API at one point
        api_.input.set_input_file_start_index(1);
        api_.input.set_input_file_end_index(nb_frames);

        // We can now launch holovibes over this file
        set_start_stop_buttons(true);

        parent_->notify();
    }
    else
        set_start_stop_buttons(false);
}

void ImportPanel::import_stop()
{
    gui::close_windows();
    api_.input.import_stop();
    parent_->notify();
}

// TODO: review function, we cannot edit UserInterfaceDescriptor here (instead of API)
void ImportPanel::import_start()
{
    gui::close_windows();
    if (api_.input.import_start())
        parent_->ui_->ImageRenderingPanel->set_computation_mode(static_cast<int>(api_.compute.get_compute_mode()));
}

void ImportPanel::update_fps() { api_.input.set_input_fps(ui_->ImportInputFpsSpinBox->value()); }

void ImportPanel::update_import_file_path()
{
    api_.input.set_input_file_path(ui_->ImportPathLineEdit->text().toStdString());
}

void ImportPanel::update_load_file_in_gpu()
{
    api_.input.set_load_file_in_gpu(ui_->LoadFileInGpuCheckBox->isChecked());
}

void ImportPanel::update_input_file_start_index()
{
    QSpinBox* start_spinbox = ui_->ImportStartIndexSpinBox;

    api_.input.set_input_file_start_index(start_spinbox->value() - 1);

    start_spinbox->setValue(static_cast<int>(api_.input.get_input_file_start_index()) + 1);
    ui_->ImportEndIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_end_index()));
}

void ImportPanel::update_input_file_end_index()
{
    QSpinBox* end_spinbox = ui_->ImportEndIndexSpinBox;

    api_.input.set_input_file_end_index(end_spinbox->value());

    end_spinbox->setValue(static_cast<int>(api_.input.get_input_file_end_index()));
    ui_->ImportStartIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_start_index()) + 1);
}

} // namespace holovibes::gui
