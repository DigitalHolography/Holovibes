/*! \file
 *
 */

#include <filesystem>

#include "export_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"
#include "GUI.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
ExportPanel::ExportPanel(QWidget* parent)
    : Panel(parent)
    , import_start_subscriber_("import_start", [this](bool success) { set_record_image_mode(); })
    , start_record_subscriber_("start_record_export_panel", [this](bool _unused) { start_record(); })
    , set_output_file_path_subscriber_("set_output_file_name",
                                       std::bind(&ExportPanel::set_output_file_name, this, std::placeholders::_1))
    , browse_record_output_file_subscriber_("browse_record_output_file",
                                            [this](bool _unused) { return browse_record_output_file().toStdString(); })
{
}

ExportPanel::~ExportPanel() {}

void ExportPanel::init()
{
    ui_->NumberOfFramesSpinBox->setSingleStep(record_frame_step_);
    set_record_mode(QString::fromUtf8("Raw Image"));
}

/*
 * \brief Small helper function NOT IN THE CLASS to update the record output file path in the UI.
 * Exists to avoid code duplication and to centralise the Notifier name 'record_output_file'.
 */
void actualise_record_output_file_ui(const std::filesystem::path file_path)
{
    NotifierManager::notify<std::filesystem::path>("record_output_file", file_path);
}

void ExportPanel::on_notify()
{
    if (api::get_record_mode() == RecordMode::CHART)
    {
        ui_->RecordExtComboBox->clear();
        ui_->RecordExtComboBox->insertItem(0, ".csv");
        ui_->RecordExtComboBox->insertItem(1, ".txt");

        ui_->ChartPlotWidget->show();

        if (gui::get_main_display())
        {
            gui::get_main_display()->resetTransform();

            gui::get_main_display()->getOverlayManager().enable<Noise>();
            gui::get_main_display()->getOverlayManager().enable<Signal>();
        }
    }
    else
    {
        if (api::get_record_mode() == RecordMode::RAW)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (api::get_record_mode() == RecordMode::HOLOGRAM)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
            ui_->RecordExtComboBox->insertItem(2, ".mp4");
        }
        else if (api::get_record_mode() == RecordMode::CUTS_YZ || api::get_record_mode() == RecordMode::CUTS_XZ)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".mp4");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
        }
        else if (api::get_record_mode() == RecordMode::MOMENTS)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }

        ui_->ChartPlotWidget->hide();

        if (gui::get_main_display())
        {
            gui::get_main_display()->resetTransform();

            gui::get_main_display()->getOverlayManager().disable(Signal);
            gui::get_main_display()->getOverlayManager().disable(Noise);
        }
    }

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
    signalBtn->setStyleSheet((gui::get_main_display() && signalBtn->isEnabled() &&
                              gui::get_main_display()->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui_->ChartNoisePushButton;
    noiseBtn->setStyleSheet((gui::get_main_display() && noiseBtn->isEnabled() &&
                             gui::get_main_display()->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(UserInterfaceDescriptor::instance().record_output_directory_) /
         UserInterfaceDescriptor::instance().output_filename_)
            .string();
    path_line_edit->insert(record_output_path.c_str());

    actualise_record_output_file_ui(record_output_path);

    if (api::get_record_frame_count().has_value())
    {
        // const QSignalBlocker blocker(ui_->NumberOfFramesSpinBox);
        ui_->NumberOfFramesSpinBox->setValue(static_cast<int>(api::get_record_frame_count().value()));
        ui_->NumberOfFramesCheckBox->setChecked(true);
        ui_->NumberOfFramesSpinBox->setEnabled(true);
    }
}

void ExportPanel::set_record_frame_step(int step)
{
    record_frame_step_ = step;
    parent_->ui_->NumberOfFramesSpinBox->setSingleStep(step);
}

int ExportPanel::get_record_frame_step() { return record_frame_step_; }

void ExportPanel::init_light_ui()
{
    actualise_record_output_file_ui(std::filesystem::path(ui_->OutputFilePathLineEdit->text().toStdString()));
}

QString ExportPanel::browse_record_output_file()
{
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    RecordMode record_mode = api::get_record_mode();

    if (record_mode == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (record_mode == RecordMode::RAW || record_mode == RecordMode::MOMENTS)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (record_mode == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }
    else if (record_mode == RecordMode::CUTS_XZ || record_mode == RecordMode::CUTS_YZ)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Mp4 files (*.mp4);; Avi Files (*.avi);;"));
    }

    if (filepath.isEmpty())
        return QString::fromStdString(api::get_record_file_path());

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    const std::string file_ext = api::browse_record_output_file(std_filepath);
    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

    parent_->notify();

    return filepath;
}

void ExportPanel::set_output_file_name(std::string std_filepath)
{
    const std::string file_ext = api::browse_record_output_file(std_filepath);
    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

    parent_->notify();
}

void ExportPanel::set_nb_frames_mode(bool value) { ui_->NumberOfFramesSpinBox->setEnabled(value); }

void ExportPanel::set_record_mode(const QString& value)
{
    if (api::get_record_mode() == RecordMode::CHART)
        stop_chart_display();

    const std::string text = value.toStdString();

    api::set_record_mode(text);

    parent_->notify();
}

void ExportPanel::stop_record() { api::stop_record(); }

void ExportPanel::record_finished(RecordMode record_mode)
{
    std::string info;

    if (record_mode == RecordMode::CHART)
        info = "Chart record finished";
    else if (record_mode == RecordMode::HOLOGRAM || record_mode == RecordMode::RAW ||
             record_mode == RecordMode::MOMENTS)
        info = "Frame record finished";

    LOG_INFO("[RECORDER] {}", info);

    ui_->RawDisplayingCheckBox->setHidden(false);
    ui_->ExportRecPushButton->setEnabled(true);
    ui_->ExportStopPushButton->setEnabled(false);
    ui_->BatchSizeSpinBox->setEnabled(api::get_compute_mode() == Computation::Hologram);

    api::record_finished();

    // notify others panels (info panel & lightUI) that the record is finished
    NotifierManager::notify<bool>("record_finished", true);
}

void ExportPanel::start_record()
{
    if (!api::start_record_preconditions()) // Check if the record can be started
        return;

    // Start record
    gui::get_raw_window().reset(nullptr);
    ui_->ViewPanel->update_raw_view(false);
    ui_->RawDisplayingCheckBox->setHidden(true);

    ui_->BatchSizeSpinBox->setEnabled(false);
    UserInterfaceDescriptor::instance().is_recording_ = true;

    // set the record progress bar color to orange, the patient should not move
    ui_->InfoPanel->set_recordProgressBar_color(QColor(209, 90, 25), "Recording: %v/%m");

    NotifierManager::notify<RecordBarColorData>("record_progress_bar_color", {QColor(209, 90, 25), "Recording"});

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);

    ui_->InfoPanel->set_visible_record_progress(true);

    auto callback = [record_mode = api::get_record_mode(), this]()
    { parent_->synchronize_thread([=]() { record_finished(record_mode); }); };

    api::start_record(callback);
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

void ExportPanel::update_record_frame_count_enabled()
{
    bool checked = ui_->NumberOfFramesCheckBox->isChecked();

    if (!checked)
        api::set_record_frame_count(std::nullopt);
    else
        api::set_record_frame_count(ui_->NumberOfFramesSpinBox->value());
}

void ExportPanel::update_record_frame_count() {}

void ExportPanel::update_record_file_path()
{
    api::set_record_file_path(ui_->OutputFilePathLineEdit->text().toStdString() +
                              ui_->RecordExtComboBox->currentText().toStdString());
}

void ExportPanel::set_record_image_mode()
{
    ui_->RecordImageModeComboBox->setCurrentText(QString("Processed Image"));
    api::set_record_mode_setting(RecordMode::HOLOGRAM);
}

/**
 * @brief called when change output file extension
 */
void ExportPanel::update_record_file_extension(const QString& value)
{
    std::string path = ui_->OutputFilePathLineEdit->text().toStdString();
    std::string ext = value.toStdString();
    api::set_record_file_path(path + ext);
}
} // namespace holovibes::gui
