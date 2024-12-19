/*! \file
 *
 */

#include <filesystem>

#include "export_panel.hh"
#include "enum_recorded_eye_type.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"

namespace holovibes::gui
{
ExportPanel::ExportPanel(QWidget* parent)
    : Panel(parent)
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
    set_record_mode(static_cast<int>(RecordMode::RAW)); // Not great but it works
}

void ExportPanel::on_notify()
{
    // File extension
    auto file_ext_view = qobject_cast<QListView*>(ui_->RecordExtComboBox->view());
    auto extension_indexes = api_.record.get_supported_formats(
        api_.record.get_record_mode()); // The indexes compatible with the current record mode
    for (int i = 0; i < ui_->RecordExtComboBox->count(); i++)
    {
        bool do_hide = std::find(extension_indexes.begin(), extension_indexes.end(), i) == extension_indexes.end();
        file_ext_view->setRowHidden(i, do_hide); // Hiding the incompatible extensions

        // Changing the current extension if it is not compatible with the current record mode
        if (i == ui_->RecordExtComboBox->currentIndex() && do_hide)
            ui_->RecordExtComboBox->setCurrentIndex(extension_indexes[0]);
    }

    if (api_.record.get_record_mode() == RecordMode::CHART)
    {
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
        ui_->ChartPlotWidget->hide();

        if (gui::get_main_display())
        {
            gui::get_main_display()->resetTransform();

            gui::get_main_display()->getOverlayManager().disable(Signal);
            gui::get_main_display()->getOverlayManager().disable(Noise);
        }
    }

    // Record type
    auto img_mode_view = qobject_cast<QListView*>(ui_->RecordImageModeComboBox->view());

    ui_->RecordImageModeComboBox->setCurrentIndex(static_cast<int>(api_.record.get_record_mode()));

    // Hiding most of the options when in raw mode
    const bool is_raw = api_.compute.get_compute_mode() == Computation::Raw;
    img_mode_view->setRowHidden(static_cast<int>(RecordMode::HOLOGRAM), is_raw);
    img_mode_view->setRowHidden(static_cast<int>(RecordMode::CHART), is_raw);
    img_mode_view->setRowHidden(static_cast<int>(RecordMode::MOMENTS), is_raw);

    // When set to raw, the 3D cuts are automatically disabled, so this works as a valid toggle for raw mode too
    const bool hide_cuts = !ui_->TimeTransformationCutsCheckBox->isChecked();
    img_mode_view->setRowHidden(static_cast<int>(RecordMode::CUTS_XZ), hide_cuts);
    img_mode_view->setRowHidden(static_cast<int>(RecordMode::CUTS_YZ), hide_cuts);

    // Chart buttons
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

    // File path
    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(UserInterfaceDescriptor::instance().record_output_directory_) /
         UserInterfaceDescriptor::instance().output_filename_)
            .string();
    path_line_edit->insert(record_output_path.c_str());
    path_line_edit->setToolTip(record_output_path.c_str());

    // Number of frames
    if (api_.record.get_record_frame_count().has_value())
    {
        // const QSignalBlocker blocker(ui_->NumberOfFramesSpinBox);
        ui_->NumberOfFramesSpinBox->setValue(static_cast<int>(api_.record.get_record_frame_count().value()));
        ui_->NumberOfFramesCheckBox->setChecked(true);
        ui_->NumberOfFramesSpinBox->setEnabled(true);
    }

    if (api_.input.get_import_type() == ImportType::File)
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeStrideSpinBox->value()));

    ui_->RecordedEyePushButton->setText(QString::fromStdString(api_.record.get_recorded_eye_display_string()));
    // Cannot disable the button because starting/stopping a recording doesn't trigger a notify
}

void ExportPanel::set_record_frame_step(int step)
{
    record_frame_step_ = step;
    parent_->ui_->NumberOfFramesSpinBox->setSingleStep(step);
}

int ExportPanel::get_record_frame_step() { return record_frame_step_; }

QString ExportPanel::browse_record_output_file()
{
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    RecordMode record_mode = api_.record.get_record_mode();

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
        return QString::fromStdString(api_.record.get_record_file_path());

    set_output_file_name(filepath.toStdString());

    return filepath;
}

void ExportPanel::set_output_file_name(std::string std_filepath)
{
    const std::string file_ext = gui::browse_record_output_file(std_filepath);
    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

    parent_->notify();
}

void ExportPanel::set_nb_frames_mode(bool value) { ui_->NumberOfFramesSpinBox->setEnabled(value); }

void ExportPanel::set_record_mode(int index)
{
    if (api_.record.get_record_mode() == RecordMode::CHART)
        stop_chart_display();

    api_.record.set_record_mode_enum(static_cast<RecordMode>(index));

    parent_->notify();
}

void ExportPanel::stop_record() { api_.record.stop_record(); }

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
    ui_->BatchSizeSpinBox->setEnabled(api_.compute.get_compute_mode() == Computation::Hologram);
    ui_->RecordedEyePushButton->setEnabled(true);

    // notify others panels (info panel & lightUI) that the record is finished
    NotifierManager::notify<bool>("record_finished", true);
}

void ExportPanel::start_record()
{
    if (!api_.record.start_record_preconditions()) // Check if the record can be started
        return;

    // Start record
    gui::get_raw_window().reset(nullptr);
    ui_->ViewPanel->update_raw_view(false);
    ui_->RawDisplayingCheckBox->setHidden(true);

    ui_->BatchSizeSpinBox->setEnabled(false);

    // set the record progress bar color to orange, the patient should not move
    ui_->InfoPanel->set_recordProgressBar_color(QColor(209, 90, 25), "Recording: %v/%m");

    NotifierManager::notify<RecordBarColorData>("record_progress_bar_color", {QColor(209, 90, 25), "Recording"});

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);
    ui_->RecordedEyePushButton->setEnabled(false);

    ui_->InfoPanel->set_visible_record_progress(true);

    auto callback = [record_mode = api_.record.get_record_mode(), this]()
    { parent_->synchronize_thread([=]() { record_finished(record_mode); }); };

    api_.record.start_record(callback);
}

void ExportPanel::activeSignalZone()
{
    gui::active_signal_zone();
    parent_->notify();
}

void ExportPanel::activeNoiseZone()
{
    gui::active_noise_zone();
    parent_->notify();
}

void ExportPanel::start_chart_display()
{
    api_.view.set_chart_display(true);
    gui::set_chart_display(true);

    connect(UserInterfaceDescriptor::instance().plot_window_.get(),
            SIGNAL(closed()),
            this,
            SLOT(stop_chart_display()),
            Qt::UniqueConnection);

    ui_->ChartPlotPushButton->setEnabled(false);
}

void ExportPanel::stop_chart_display()
{
    api_.view.set_chart_display(false);
    gui::set_chart_display(false);

    ui_->ChartPlotPushButton->setEnabled(true);
}

void ExportPanel::update_record_frame_count_enabled()
{
    bool checked = ui_->NumberOfFramesCheckBox->isChecked();

    if (!checked)
        api_.record.set_record_frame_count(std::nullopt);
    else
        api_.record.set_record_frame_count(ui_->NumberOfFramesSpinBox->value());
}

void ExportPanel::update_record_file_path()
{
    api_.record.set_record_file_path(ui_->OutputFilePathLineEdit->text().toStdString() +
                                     ui_->RecordExtComboBox->currentText().toStdString());
}

/**
 * @brief called when change output file extension
 */
void ExportPanel::update_record_file_extension(const QString& value)
{
    std::string path = ui_->OutputFilePathLineEdit->text().toStdString();
    std::string ext = value.toStdString();
    api_.record.set_record_file_path(path + ext);
}

void ExportPanel::update_recorded_eye()
{
    api_.record.set_recorded_eye(api_.record.get_recorded_eye() == RecordedEyeType::LEFT ? RecordedEyeType::RIGHT
                                                                                         : RecordedEyeType::LEFT);
    on_notify();
}

void ExportPanel::reset_recorded_eye()
{
    api_.record.set_recorded_eye(RecordedEyeType::NONE);
    on_notify();
}

} // namespace holovibes::gui
