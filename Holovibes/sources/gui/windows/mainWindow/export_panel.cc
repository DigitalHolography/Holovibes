#include <filesystem>

#include "export_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"

namespace holovibes::gui
{
ExportPanel::ExportPanel(QWidget* parent)
    : Panel(parent)
{
    default_output_filename_ = "capture";
    record_output_directory_ = (get_user_documents_path() / "Holovibes").string();
    batch_input_directory_ = "C:\\";
}

ExportPanel::~ExportPanel() {}

void ExportPanel::init() { set_record_mode(QString::fromUtf8("Raw Image")); }

void ExportPanel::on_notify()
{
    if (parent_->is_raw_mode())
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

    QPushButton* signalBtn = ui_->ChartSignalPushButton;
    signalBtn->setStyleSheet((parent_->mainDisplay && signalBtn->isEnabled() &&
                              parent_->mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui_->ChartNoisePushButton;
    noiseBtn->setStyleSheet((parent_->mainDisplay && noiseBtn->isEnabled() &&
                             parent_->mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    QLineEdit* path_line_edit = ui_->OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(record_output_directory_) / default_output_filename_).string();
    path_line_edit->insert(record_output_path.c_str());
}

void ExportPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    default_output_filename_ = ptree.get<std::string>("files.default_output_filename", default_output_filename_);
    record_output_directory_ = ptree.get<std::string>("files.record_output_directory", record_output_directory_);
    batch_input_directory_ = ptree.get<std::string>("files.batch_input_directory", batch_input_directory_);
    record_frame_step_ = ptree.get<uint>("record.record_frame_step", record_frame_step_);
}

void ExportPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<std::string>("files.default_output_filename", default_output_filename_);
    ptree.put<std::string>("files.record_output_directory", record_output_directory_);
    ptree.put<std::string>("files.batch_input_directory", batch_input_directory_);
    ptree.put<uint>("record.record_frame_step", record_frame_step_);
}

void ExportPanel::browse_record_output_file()
{
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    if (record_mode_ == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (record_mode_ == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (record_mode_ == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }

    if (filepath.isEmpty())
        return;

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    record_output_directory_ = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    default_output_filename_ = path.stem().string();

    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui_->RecordExtComboBox->setCurrentText(file_ext.c_str());

    parent_->notify();
}

void ExportPanel::set_nb_frames_mode(bool value) { ui_->NumberOfFramesSpinBox->setEnabled(value); }

void ExportPanel::browse_batch_input()
{

    // Open file explorer on the fly
    QString filename =
        QFileDialog::getOpenFileName(this, tr("Batch input file"), batch_input_directory_.c_str(), tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui_->BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

void ExportPanel::set_record_mode(const QString& value)
{
    if (record_mode_ == RecordMode::CHART)
        stop_chart_display();

    stop_record();

    const std::string text = value.toStdString();

    if (text == "Chart")
        record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        record_mode_ = RecordMode::RAW;
    else
        throw std::exception("Record mode not handled");

    if (record_mode_ == RecordMode::CHART)
    {
        ui_->RecordExtComboBox->clear();
        ui_->RecordExtComboBox->insertItem(0, ".csv");
        ui_->RecordExtComboBox->insertItem(1, ".txt");

        ui_->ChartPlotWidget->show();

        if (parent_->mainDisplay)
        {
            parent_->mainDisplay->resetTransform();

            parent_->mainDisplay->getOverlayManager().enable_all(Signal);
            parent_->mainDisplay->getOverlayManager().enable_all(Noise);
            parent_->mainDisplay->getOverlayManager().create_overlay<Signal>();
        }
    }
    else
    {
        if (record_mode_ == RecordMode::RAW)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (record_mode_ == RecordMode::HOLOGRAM)
        {
            ui_->RecordExtComboBox->clear();
            ui_->RecordExtComboBox->insertItem(0, ".holo");
            ui_->RecordExtComboBox->insertItem(1, ".avi");
            ui_->RecordExtComboBox->insertItem(2, ".mp4");
        }

        ui_->ChartPlotWidget->hide();

        if (parent_->mainDisplay)
        {
            parent_->mainDisplay->resetTransform();

            parent_->mainDisplay->getOverlayManager().disable_all(Signal);
            parent_->mainDisplay->getOverlayManager().disable_all(Noise);
        }
    }

    parent_->notify();
}

void ExportPanel::stop_record()
{
    parent_->holovibes_.stop_batch_gpib();

    if (record_mode_ == RecordMode::CHART)
        parent_->holovibes_.stop_chart_record();
    else if (record_mode_ == RecordMode::HOLOGRAM || record_mode_ == RecordMode::RAW)
        parent_->holovibes_.stop_frame_record();
}

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

    LOG_INFO << "[RECORDER] " << info;

    ui_->RawDisplayingCheckBox->setHidden(false);
    ui_->ExportRecPushButton->setEnabled(true);
    ui_->ExportStopPushButton->setEnabled(false);
    ui_->BatchSizeSpinBox->setEnabled(parent_->cd_.compute_mode == Computation::Hologram);
    is_recording_ = false;
}

void ExportPanel::start_record()
{
    bool batch_enabled = ui_->BatchGroupBox->isChecked();

    // Preconditions to start record

    std::optional<unsigned int> nb_frames_to_record = ui_->NumberOfFramesSpinBox->value();
    if (!ui_->NumberOfFramesCheckBox->isChecked())
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || record_mode_ == RecordMode::CHART) && nb_frames_to_record == std::nullopt)
    {
        LOG_ERROR << "Number of frames must be activated";
        return;
    }

    std::string output_path =
        ui_->OutputFilePathLineEdit->text().toStdString() + ui_->RecordExtComboBox->currentText().toStdString();

    std::string batch_input_path = ui_->BatchInputPathLineEdit->text().toStdString();
    if (batch_enabled && batch_input_path.empty())
    {
        LOG_ERROR << "No batch input file";
        return;
    }

    // Start record
    parent_->raw_window.reset(nullptr);
    ui_->ViewPanel->disable_raw_view();
    ui_->RawDisplayingCheckBox->setHidden(true);

    ui_->BatchSizeSpinBox->setEnabled(false);
    is_recording_ = true;

    ui_->ExportRecPushButton->setEnabled(false);
    ui_->ExportStopPushButton->setEnabled(true);

    ui_->InfoPanel->set_visible_record_progress(true);

    auto callback = [record_mode = record_mode_, this]() {
        parent_->synchronize_thread([=]() { record_finished(record_mode); });
    };

    if (batch_enabled)
    {
        parent_->holovibes_.start_batch_gpib(batch_input_path,
                                             output_path,
                                             nb_frames_to_record.value(),
                                             record_mode_,
                                             callback);
    }
    else
    {
        if (record_mode_ == RecordMode::CHART)
        {
            parent_->holovibes_.start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else if (record_mode_ == RecordMode::HOLOGRAM)
        {
            parent_->holovibes_.start_frame_record(output_path, nb_frames_to_record, false, 0, callback);
        }
        else if (record_mode_ == RecordMode::RAW)
        {
            parent_->holovibes_.start_frame_record(output_path, nb_frames_to_record, true, 0, callback);
        }
    }
}

void ExportPanel::activeSignalZone()
{
    parent_->mainDisplay->getOverlayManager().create_overlay<Signal>();
    parent_->notify();
}

void ExportPanel::activeNoiseZone()
{
    parent_->mainDisplay->getOverlayManager().create_overlay<Noise>();
    parent_->notify();
}

void ExportPanel::start_chart_display()
{
    if (parent_->cd_.chart_display_enabled)
        return;

    auto pipe = parent_->holovibes_.get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    parent_->plot_window_ =
        std::make_unique<PlotWindow>(*parent_->holovibes_.get_compute_pipe()->get_chart_display_queue(),
                                     parent_->auto_scale_point_threshold_,
                                     "Chart");
    connect(parent_->plot_window_.get(), SIGNAL(closed()), this, SLOT(stop_chart_display()), Qt::UniqueConnection);

    ui_->ChartPlotPushButton->setEnabled(false);
}

void ExportPanel::stop_chart_display()
{
    if (!parent_->cd_.chart_display_enabled)
        return;

    try
    {
        auto pipe = parent_->holovibes_.get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    parent_->plot_window_.reset(nullptr);

    ui_->ChartPlotPushButton->setEnabled(true);
}
} // namespace holovibes::gui
