/*! \file
 *
 */
#include "import_panel.hh"

#include "API.hh"
#include "GUI.hh"
#include "MainWindow.hh"
#include "user_interface_descriptor.hh"

namespace holovibes::gui
{
ImportPanel::ImportPanel(QWidget* parent)
    : Panel(parent)
{
}

ImportPanel::~ImportPanel() {}

void ImportPanel::on_notify()
{
    ui_->ImportStartIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_start_index() + 1));
    ui_->ImportEndIndexSpinBox->setValue(static_cast<int>(api_.input.get_input_file_end_index()));
    const char step = api_.input.get_data_type() == RecordedDataType::MOMENTS ? 3 : 1;
    ui_->ImportStartIndexSpinBox->setSingleStep(step);
    ui_->ImportEndIndexSpinBox->setSingleStep(step);

    const bool comp = !api_.compute.get_is_computation_stopped();
    ui_->FileReaderProgressBar->setVisible(comp && api_.input.get_import_type() == ImportType::File);

    ui_->FileLoadKindComboBox->setCurrentIndex(static_cast<int>(api_.input.get_file_load_kind()));
}

void ImportPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, ui_->ImportExportFrame->isHidden(), "panels", "import export hidden");
    ui_->actionImportExport->setChecked(!h);
    ui_->ImportExportFrame->setHidden(h);

    ui_->ImportInputFpsSpinBox->setValue(json_get_or_default(j_us, 10000, "import", "fps"));
    update_fps(); // Required as it is called `OnEditedFinished` only.

    ui_->FileLoadKindComboBox->setCurrentIndex(json_get_or_default(j_us, 0, "import", "load file kind"));
}

void ImportPanel::save_gui(json& j_us)
{
    j_us["panels"]["import export hidden"] = ui_->ImportExportFrame->isHidden();

    j_us["import"]["fps"] = ui_->ImportInputFpsSpinBox->value();
    j_us["import"]["load file kind"] = ui_->FileLoadKindComboBox->currentIndex();
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
    QLineEdit* import_line_edit = ui_->ImportPathLineEdit;

    // Insert the newly getted path in it
    import_line_edit->clear();
    import_line_edit->insert(filename);

    // Import file will then be called since it will trigger the signal `textChanged` of the line edit
}

void ImportPanel::import_file(const QString& filename)
{
    if (filename.isEmpty())
        return;

    // Start importing the chosen
    gui::close_windows();
    std::optional<io_files::InputFrameFile*> input_file_opt = api_.input.import_file(filename.toStdString());

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();

        // Gather data from the newly opened file
        int nb_frames = static_cast<int>(input_file->get_total_nb_frames());

        // Don't need the input file anymore
        delete input_file;

        // The start index cannot exceed the end index
        ui_->ImportStartIndexSpinBox->setMaximum(nb_frames);
        ui_->ImportEndIndexSpinBox->setMaximum(nb_frames);

        // We can now launch holovibes over this file
        set_start_stop_buttons(true);

        parent_->notify();
    }
    else
        set_start_stop_buttons(false);
}

void ImportPanel::import_stop() { gui::stop(); }

void ImportPanel::import_start() { gui::start(parent_->window_max_size); }

void ImportPanel::update_fps() { api_.input.set_input_fps(ui_->ImportInputFpsSpinBox->value()); }

void ImportPanel::update_import_file_path() { import_file(ui_->ImportPathLineEdit->text()); }

void ImportPanel::update_file_load_kind(int kind) { api_.input.set_file_load_kind(static_cast<FileLoadKind>(kind)); }

void ImportPanel::update_input_file_start_index()
{
    api_.input.set_input_file_start_index(ui_->ImportStartIndexSpinBox->value() - 1);
    on_notify();
}

void ImportPanel::update_input_file_end_index()
{
    api_.input.set_input_file_end_index(ui_->ImportEndIndexSpinBox->value());
    on_notify();
}

} // namespace holovibes::gui
