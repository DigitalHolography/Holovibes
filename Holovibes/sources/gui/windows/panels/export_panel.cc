#include <filesystem>

#include "export_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ExportPanel::ExportPanel(QWidget* parent)
    : Panel(parent)
{
}

ExportPanel::~ExportPanel() {}

void ExportPanel::init()
{
    ui_->NumberOfFramesSpinBox->setSingleStep(record_frame_step_);
    set_record_mode(QString::fromUtf8("Raw Image"));
}

void ExportPanel::on_notify()
{
    if (api::get_compute_mode() == Computation::Raw)
    {
        ui_->RecordImageModeComboBox->removeItem(ui_->RecordImageModeComboBox->findText("Processed Image"));
        ui_->RecordImageModeComboBox->removeItem(ui_->RecordImageModeComboBox->findText("Chart"));
    }
    else // Hologram mode
    {
        if (ui_->RecordImageModeComboBox->findText("Processed Image") == -1)
            ui_->RecordImageModeComboBox->insertItem(1, "Processed Image");
        if (ui_->RecordImageModeComboBox->findText("Chart") == -1)
            ui_->RecordImageModeComboBox->insertItem(2, "Chart");
    }

    if (ui_->TimeTransformationCutsCheckBox->isChecked())
    {
        // Only one check is needed
        if (ui_->RecordImageModeComboBox->findText("3D Cuts XZ") == -1)
        {
            ui_->RecordImageModeComboBox->insertItem(1, "3D Cuts XZ");
            ui_->RecordImageModeComboBox->insertItem(1, "3D Cuts YZ");
        }
    }
    else
    {
        ui_->RecordImageModeComboBox->removeItem(ui_->RecordImageModeComboBox->findText("3D Cuts XZ"));
        ui_->RecordImageModeComboBox->removeItem(ui_->RecordImageModeComboBox->findText("3D Cuts YZ"));
    }

    QPushButton* signalBtn = ui_->ChartSignalPushButton;
    signalBtn->setStyleSheet((api::get_main_display() && signalBtn->isEnabled() &&
                              api::get_main_display()->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui_->ChartNoisePushButton;
    noiseBtn->setStyleSheet((api::get_main_display() && noiseBtn->isEnabled() &&
                             api::get_main_display()->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(UserInterfaceDescriptor::instance().record_output_directory_) /
         UserInterfaceDescriptor::instance().default_output_filename_)
            .string();
    path_line_edit->insert(record_output_path.c_str());
}

void ExportPanel::set_record_frame_step(int step)
{
    record_frame_step_ = step;
    parent_->ui_->NumberOfFramesSpinBox->setSingleStep(step);
}

int ExportPanel::get_record_frame_step() { return record_frame_step_; }

void ExportPanel::browse_record_output_file()
{
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_XZ ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_YZ)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Mp4 files (*.mp4);; Avi Files (*.avi);;"));
    }

    if (filepath.isEmpty())
        return;

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    const std::string file_ext = api::browse_record_output_file(std_filepath);
    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

    parent_->notify();
}

void ExportPanel::set_nb_frames_mode(bool value) { ui_->NumberOfFramesSpinBox->setEnabled(value); }

void ExportPanel::browse_batch_input()
{

    // Open file explorer on the fly
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("Batch input file"),
                                                    UserInterfaceDescriptor::instance().batch_input_directory_.c_str(),
                                                    tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui_->BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

void ExportPanel::set_record_mode(const QString& value)
{
    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        stop_chart_display();

    stop_record();

    const std::string text = value.toStdString();

    api::set_record_mode(text);

    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
    {
        ui_->RecordExtComboBox->clear();
        ui_->RecordExtComboBox->insertItem(0, ".csv");
        ui_->RecordExtComboBox->insertItem(1, ".txt");

        ui_->ChartPlotWidget->show();

        if (api::get_main_display())
        {
            api::get_main_display()->resetTransform();

            api::get_main_display()->getOverlayManager().enable_all(Signal);
            api::get_main_display()->getOverlayManager().enable_all(Noise);
            api::get_main_display()->getOverlayManager().create_overlay<Signal>();
        }
    }
    else
    {
        if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
            ui_->RecordExtComboBox->insertItem(2, ".mp4");
        }
        else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_YZ ||
                 UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_XZ)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".mp4");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
        }

        ui_->ChartPlotWidget->hide();

        if (api::get_main_display())
        {
            api::get_main_display()->resetTransform();

            api::get_main_display()->getOverlayManager().disable_all(Signal);
            api::get_main_display()->getOverlayManager().disable_all(Noise);
        }
    }

    parent_->notify();
}

void ExportPanel::stop_record() { api::stop_record(); }

void ExportPanel::record_finished(RecordMode record_mode)
{
    std::string info;

    if (record_mode == RecordMode::CHART)
        info = "Chart record finished";
    else if (record_mode == RecordMode::HOLOGRAM || record_mode == RecordMode::RAW)
        info = "Frame record finished";

    ui_->InfoPanel->set_visible_record_progress(false);

    if (ui_->BatchGroupBox->isChecked())
        info = "Batch " + info;

    LOG_INFO(main, "[RECORDER] {}", info);

    ui_->RawDisplayingCheckBox->setHidden(false);
    ui_->ExportRecPushButton->setEnabled(true);
    ui_->ExportStopPushButton->setEnabled(false);
    ui_->BatchSizeSpinBox->setEnabled(api::get_compute_mode() == Computation::Hologram);
    api::record_finished();
}

void ExportPanel::start_record()
{
    bool batch_enabled = ui_->BatchGroupBox->isChecked();
    bool nb_frame_checked = ui_->NumberOfFramesCheckBox->isChecked();
    std::optional<unsigned int> nb_frames_to_record = std::nullopt;

    if (nb_frame_checked)
        nb_frames_to_record = ui_->NumberOfFramesSpinBox->value();

    std::string output_path =
        ui_->OutputFilePathLineEdit->text().toStdString() + ui_->RecordExtComboBox->currentText().toStdString();
    std::string batch_input_path = ui_->BatchInputPathLineEdit->text().toStdString();

    // Preconditions to start record
    const bool preconditions =
        api::start_record_preconditions(batch_enabled, nb_frame_checked, nb_frames_to_record, batch_input_path);

    if (!preconditions)
        return;

    // Start record
    api::get_raw_window().reset(nullptr);
    ui_->ViewPanel->update_raw_view(false);
    ui_->RawDisplayingCheckBox->setHidden(true);

    ui_->BatchSizeSpinBox->setEnabled(false);
    UserInterfaceDescriptor::instance().is_recording_ = true;

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);

    ui_->InfoPanel->set_visible_record_progress(true);

    auto callback = [record_mode = UserInterfaceDescriptor::instance().record_mode_, this]()
    { parent_->synchronize_thread([=]() { record_finished(record_mode); }); };

    api::start_record(batch_enabled, nb_frames_to_record, output_path, batch_input_path, callback);
}

void ExportPanel::activeSignalZone()
{
    api::active_signal_zone();
    parent_->notify();
}

void ExportPanel::activeNoiseZone()
{
    api::active_noise_zone();
    parent_->notify();
}

void ExportPanel::start_chart_display()
{
    if (api::get_chart_display_enabled())
        return;

    api::start_chart_display();
    connect(UserInterfaceDescriptor::instance().plot_window_.get(),
            SIGNAL(closed()),
            this,
            SLOT(stop_chart_display()),
            Qt::UniqueConnection);

    ui_->ChartPlotPushButton->setEnabled(false);
}

void ExportPanel::stop_chart_display()
{
    if (!api::get_chart_display_enabled())
        return;

    api::stop_chart_display();

    ui_->ChartPlotPushButton->setEnabled(true);
}
} // namespace holovibes::gui
