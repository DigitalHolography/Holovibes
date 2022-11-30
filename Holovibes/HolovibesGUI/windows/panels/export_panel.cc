/*! \file
 *
 */

#include <filesystem>

#include "export_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"

#include "user_interface.hh"

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
    if (api::get_compute_mode() == ComputeModeEnum::Raw)
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
    signalBtn->setStyleSheet((UserInterface::instance().xy_window && signalBtn->isEnabled() &&
                              UserInterface::instance().xy_window->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui_->ChartNoisePushButton;
    noiseBtn->setStyleSheet((UserInterface::instance().xy_window && noiseBtn->isEnabled() &&
                             UserInterface::instance().xy_window->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path = (std::filesystem::path(UserInterface::instance().record_output_directory_) /
                                      UserInterface::instance().default_output_filename_)
                                         .string();
    path_line_edit->insert(record_output_path.c_str());

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);
    ui_->ChartPlotPushButton->setEnabled(api::detail::get_value<ChartDisplayEnabled>() == false);
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
    if (api::get_chart_record().is_running)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterface::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }
    else if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::CUTS_XZ ||
             api::get_frame_record().record_type == FrameRecordStruct::RecordType::CUTS_YZ)
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

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    UserInterface::instance().record_output_directory_ = path.parent_path().string();
    UserInterface::instance().default_output_filename_ = path.stem().string();

    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(path.extension().string().c_str());

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
    LOG_FUNC(text);

    // FIXME API : Maybe a map

    // TODO: Dictionnary
    if (text == "Chart")
        api::change_chart_record()->is_selected = true;
    else
        api::change_chart_record()->is_selected = false;

    if (text == "Processed Image")
        api::change_frame_record()->record_type = FrameRecordStruct::RecordType::HOLOGRAM;
    else if (text == "Raw Image")
        api::change_frame_record()->record_type = FrameRecordStruct::RecordType::RAW;
    else if (text == "3D Cuts XZ")
        api::change_frame_record()->record_type = FrameRecordStruct::RecordType::CUTS_XZ;
    else if (text == "3D Cuts YZ")
        api::change_frame_record()->record_type = FrameRecordStruct::RecordType::CUTS_YZ;
    else if (text != "Chart")
        throw std::exception("Record mode not handled");
}

void ExportPanel::set_frame_record_mode(const QString& value)
{
    if (api::get_chart_record().is_running)
        stop_chart_display();

    stop_record();

    set_record_mode_in_api_from_text(value.toStdString());

    if (api::get_chart_record().is_running)
    {
        ui_->RecordExtComboBox->clear();
        ui_->RecordExtComboBox->insertItem(0, ".csv");
        ui_->RecordExtComboBox->insertItem(1, ".txt");

        ui_->ChartPlotWidget->show();

        if (UserInterface::instance().xy_window)
        {
            UserInterface::instance().xy_window->resetTransform();

            UserInterface::instance().xy_window->getOverlayManager().enable_all(KindOfOverlay::Signal);
            UserInterface::instance().xy_window->getOverlayManager().enable_all(KindOfOverlay::Noise);
            UserInterface::instance().xy_window->getOverlayManager().create_overlay<KindOfOverlay::Signal>();
        }
    }
    else
    {
        if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::RAW)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::HOLOGRAM)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
            ui_->RecordExtComboBox->insertItem(2, ".mp4");
        }
        else if (api::get_frame_record().record_type == FrameRecordStruct::RecordType::CUTS_YZ ||
                 api::get_frame_record().record_type == FrameRecordStruct::RecordType::CUTS_XZ)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".mp4");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
        }

        ui_->ChartPlotWidget->hide();

        if (UserInterface::instance().xy_window)
        {
            UserInterface::instance().xy_window->resetTransform();

            UserInterface::instance().xy_window->getOverlayManager().disable_all(KindOfOverlay::Signal);
            UserInterface::instance().xy_window->getOverlayManager().disable_all(KindOfOverlay::Noise);
        }
    }

    parent_->notify();
}

void ExportPanel::stop_record()
{
    api::change_chart_record()->is_running = false;
    api::change_frame_record()->is_running = false;
}

void ExportPanel::record_finished(FrameRecordStruct::RecordType record_mode)
{
    std::string info;

    if (api::get_chart_record().is_running)
        info = "Chart record finished";
    else if (api::get_frame_record().is_running)
        info = "Frame record finished";

    if (ui_->BatchGroupBox->isChecked())
        info = "Batch " + info;

    LOG_INFO("[RECORDER] {}", info);
}

void ExportPanel::start_record()
{
    if (ui_->BatchGroupBox->isChecked())
        api::set_script_path(ui_->BatchInputPathLineEdit->text().toStdString());
    else
        api::set_script_path("");

    if (ui_->NumberOfFramesCheckBox->isChecked())
    {
        api::change_chart_record()->nb_points_to_record = ui_->NumberOfFramesSpinBox->value();
        api::change_frame_record()->nb_frames_to_record = ui_->NumberOfFramesSpinBox->value();
    }
    else
    {
        api::change_chart_record()->nb_points_to_record = 0;
        api::change_frame_record()->nb_frames_to_record = 0;
    }

    std::string output_path =
        ui_->OutputFilePathLineEdit->text().toStdString() + ui_->RecordExtComboBox->currentText().toStdString();

    api::change_chart_record()->chart_file_path = output_path;
    api::change_frame_record()->frames_file_path = output_path;

    // FIXME API --- FIXME RECORD
}

void ExportPanel::activeSignalZone()
{
    UserInterface::instance().xy_window->getOverlayManager().create_overlay<gui::KindOfOverlay::Signal>();
}

void ExportPanel::activeNoiseZone()
{
    UserInterface::instance().xy_window->getOverlayManager().create_overlay<gui::KindOfOverlay::Noise>();
}

void ExportPanel::start_chart_display() { api::detail::set_value<ChartDisplayEnabled>(true); }

void ExportPanel::stop_chart_display() { api::detail::set_value<ChartDisplayEnabled>(false); }
} // namespace holovibes::gui
