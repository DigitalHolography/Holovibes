/*! \file
 *
 */

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
    UserInterface::instance().export_panel = this;
}

ExportPanel::~ExportPanel() {}

void ExportPanel::init()
{
    ui_->NumberOfFramesSpinBox->setSingleStep(record_frame_step_);
    set_frame_record_mode(QString::fromUtf8("Raw Image"));
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
    signalBtn->setStyleSheet((UserInterface::instance().main_display && signalBtn->isEnabled() &&
                              UserInterface::instance().main_display->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui_->ChartNoisePushButton;
    noiseBtn->setStyleSheet((UserInterface::instance().main_display && noiseBtn->isEnabled() &&
                             UserInterface::instance().main_display->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path = (std::filesystem::path(UserInterface::instance().record_output_directory_) /
                                      UserInterface::instance().default_output_filename_)
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
    if (api::get_frame_record_mode().record_mode == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (api::get_frame_record_mode().record_mode == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (api::get_frame_record_mode().record_mode == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }
    else if (api::get_frame_record_mode().record_mode == RecordMode::CUTS_XZ ||
             api::get_frame_record_mode().record_mode == RecordMode::CUTS_YZ)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
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
                                                    UserInterface::instance().batch_input_directory_.c_str(),
                                                    tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui_->BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

static void set_record_mode_in_api_from_text(const std::string& text)
{
    LOG_FUNC(main, text);

    // FIXME API : Maybe a map

    // TODO: Dictionnary
    if (text == "Chart")
        api::change_frame_record_mode()->record_mode = RecordMode::CHART;
    else if (text == "Processed Image")
        api::change_frame_record_mode()->record_mode = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        api::change_frame_record_mode()->record_mode = RecordMode::RAW;
    else if (text == "3D Cuts XZ")
        api::change_frame_record_mode()->record_mode = RecordMode::CUTS_XZ;
    else if (text == "3D Cuts YZ")
        api::change_frame_record_mode()->record_mode = RecordMode::CUTS_YZ;
    else
        throw std::exception("Record mode not handled");
}

void ExportPanel::set_frame_record_mode(const QString& value)
{
    if (api::get_frame_record_mode().record_mode == RecordMode::CHART)
        stop_chart_display();

    stop_record();

    set_record_mode_in_api_from_text(value.toStdString());

    if (api::get_frame_record_mode().record_mode == RecordMode::CHART)
    {
        ui_->RecordExtComboBox->clear();
        ui_->RecordExtComboBox->insertItem(0, ".csv");
        ui_->RecordExtComboBox->insertItem(1, ".txt");

        ui_->ChartPlotWidget->show();

        if (UserInterface::instance().main_display)
        {
            UserInterface::instance().main_display->resetTransform();

            UserInterface::instance().main_display->getOverlayManager().enable_all(KindOfOverlay::Signal);
            UserInterface::instance().main_display->getOverlayManager().enable_all(KindOfOverlay::Noise);
            UserInterface::instance().main_display->getOverlayManager().create_overlay<KindOfOverlay::Signal>();
        }
    }
    else
    {
        if (api::get_frame_record_mode().record_mode == RecordMode::RAW)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (api::get_frame_record_mode().record_mode == RecordMode::HOLOGRAM)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
            ui_->RecordExtComboBox->insertItem(2, ".mp4");
        }
        else if (api::get_frame_record_mode().record_mode == RecordMode::CUTS_YZ ||
                 api::get_frame_record_mode().record_mode == RecordMode::CUTS_XZ)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".mp4");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
        }

        ui_->ChartPlotWidget->hide();

        if (UserInterface::instance().main_display)
        {
            UserInterface::instance().main_display->resetTransform();

            UserInterface::instance().main_display->getOverlayManager().disable_all(KindOfOverlay::Signal);
            UserInterface::instance().main_display->getOverlayManager().disable_all(KindOfOverlay::Noise);
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

    LOG_INFO("[RECORDER] {}", info);

    ui_->RawDisplayingCheckBox->setHidden(false);
    ui_->ExportRecPushButton->setEnabled(true);
    ui_->ExportStopPushButton->setEnabled(false);
    ui_->BatchSizeSpinBox->setEnabled(api::get_compute_mode() == Computation::Hologram);
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
    UserInterface()::instance().raw_window->reset(nullptr);
    ui_->ViewPanel->update_raw_view(false);
    ui_->RawDisplayingCheckBox->setHidden(true);

    ui_->BatchSizeSpinBox->setEnabled(false);

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);

    ui_->InfoPanel->set_visible_record_progress(true);

    auto callback = [record_mode = api::get_frame_record_mode().record_mode, this]()
    { parent_->synchronize_thread([=]() { record_finished(record_mode); }); };

    api::start_record(batch_enabled, nb_frames_to_record, output_path, batch_input_path, callback);
}

void ExportPanel::activeSignalZone()
{
    UserInterface::instance().main_display->getOverlayManager().create_overlay<gui::KindOfOverlay::Signal>();
}

void ExportPanel::activeNoiseZone()
{
    UserInterface::instance().main_display->getOverlayManager().create_overlay<gui::KindOfOverlay::Noise>();
}

void ExportPanel::start_chart_display() { api::detail::set_value<ChartDisplayEnabled>(true); }

void ExportPanel::stop_chart_display() { api::detail::set_value<ChartDisplayEnabled>(false); }
} // namespace holovibes::gui
