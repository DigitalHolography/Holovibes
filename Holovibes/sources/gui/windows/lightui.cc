#include <filesystem>
#include <cmath> // Pour std::round

#include "lightui.hh"
#include "MainWindow.hh"
#include "export_panel.hh"
#include "image_rendering_panel.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"
#include "GUI.hh"

#pragma warning(push, 0)
#include "ui_lightui.h"
#pragma warning(pop)

#include "notifier.hh"

namespace holovibes::gui
{
LightUI::LightUI(QWidget* parent, MainWindow* main_window)
    : QMainWindow(parent)
    , ui_(new Ui::LightUI)
    , main_window_(main_window)
    , visible_(false)
    , notify_subscriber_("notify", std::bind(&LightUI::on_notify, this, std::placeholders::_1))
    , record_start_subscriber_("record_start", std::bind(&LightUI::on_record_start, this, std::placeholders::_1))
    , record_end_subscriber_("record_stop", std::bind(&LightUI::on_record_stop, this, std::placeholders::_1))
    , record_progress_subscriber_("record_progress",
                                  std::bind(&LightUI::on_record_progress, this, std::placeholders::_1))
    , record_progress_bar_color_subscriber_(
          "record_progress_bar_color", std::bind(&LightUI::on_record_progress_bar_color, this, std::placeholders::_1))
    , record_finished_subscriber_("record_finished",
                                  [this](bool success)
                                  {
                                      reset_start_button();
                                      reset_record_progress_bar();
                                  })
{
    ui_->setupUi(this);
}

LightUI::~LightUI()
{
    gui::set_light_ui_mode(visible_);

    delete ui_;
}

void LightUI::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    visible_ = true;
    notify();
}

void LightUI::z_value_changed(int z_distance)
{
    API.transform.set_z_distance(static_cast<float>(z_distance) / 1000.0f);

    // The slider and the box must have the same value
    ui_->ZSpinBox->setValue(static_cast<int>(std::round(z_distance)));
    ui_->ZSlider->setValue(static_cast<int>(std::round(z_distance)));
}

void LightUI::browse_record_output_file_ui()
{
    // FIXED: This is a kind of hack that works with the current implementation of MainWindow. Ideally, lightui should
    // not know about the MainWindow and the ExportPanel. It should only know about the API. One way to fix it is to
    // create a new browser in this class and then use notify to send the file path to the API (and synchronize the API
    // with the file path).
    //? FIXED: The notifier system is now in use instead ; it also features (optional) return types.

    std::filesystem::path file_path{NotifierManager::notify<bool, std::string>("browse_record_output_file", true)};
    std::string file_path_str = file_path.string();
    std::replace(file_path_str.begin(), file_path_str.end(), '/', '\\');
    ui_->OutputFilePathLineEdit->setText(QString::fromStdString(file_path_str));

    // remove the extension from the filename
    ui_->OutputFileNameLineEdit->setText(QString::fromStdString(file_path.stem().string()));
}
void LightUI::set_record_file_name()
{
    QString filename = ui_->OutputFileNameLineEdit->text();
    // concatenate the path with the filename
    std::filesystem::path path(ui_->OutputFilePathLineEdit->text().toStdString());
    std::filesystem::path file(filename.toStdString());

    NotifierManager::notify<std::string>("set_output_file_name", (path / file).string());
}

void LightUI::start_stop_recording(bool start)
{
    if (start)
        NotifierManager::notify<bool>("start_record_export_panel", true);
    else
        API.record.stop_record();
}

void LightUI::on_record_start(RecordMode record)
{
    ui_->startButton->setText("Stop recording");
    ui_->RecordedEyePushButton->setEnabled(false);
    ui_->ResetRecordedEyePushButton->setEnabled(false);
    LOG_INFO("Recording started");
}

void LightUI::on_record_stop(RecordMode record)
{
    reset_start_button();

    ui_->RecordedEyePushButton->setEnabled(true);
    ui_->ResetRecordedEyePushButton->setEnabled(true);

    reset_record_progress_bar();

    LOG_INFO("Recording stopped");
}

void LightUI::on_record_progress(const RecordProgressData& data) { actualise_record_progress(data.value, data.max); }

void LightUI::on_record_progress_bar_color(const RecordBarColorData& data)
{
    set_recordProgressBar_color(data.color, data.text);
}

void LightUI::reset_start_button()
{
    ui_->startButton->setChecked(false);
    ui_->startButton->setText("Start recording");
}

void LightUI::actualise_record_progress(const int value, const int max)
{
    ui_->recordProgressBar->setMaximum(max);
    ui_->recordProgressBar->setValue(value);
}

void LightUI::reset_record_progress_bar()
{
    set_recordProgressBar_color(QColor(10, 10, 10), "Idle");
    actualise_record_progress(0, 1); // So as to reset the progress of the bar.
}

void LightUI::notify()
{
    // Z distance
    auto& api = API;
    float z_distance = api.transform.get_z_distance();

    ui_->ZSpinBox->setValue(static_cast<int>(std::round(z_distance * 1000)));
    ui_->ZSlider->setValue(static_cast<int>(std::round(z_distance * 1000)));

    // Filename
    std::filesystem::path file_path{API.record.get_record_file_path()};
    ui_->OutputFilePathLineEdit->setText(QString::fromStdString(file_path.parent_path().string()));
    // remove the extension from the filename
    ui_->OutputFileNameLineEdit->setText(QString::fromStdString(file_path.stem().string()));

    // Contrast
    bool pipe_loaded = api.compute.get_compute_pipe_no_throw() != nullptr;
    ui_->ContrastCheckBox->setChecked(pipe_loaded && api.contrast.get_contrast_enabled());
    ui_->ContrastCheckBox->setEnabled(pipe_loaded);
    ui_->AutoRefreshContrastCheckBox->setChecked(api.contrast.get_contrast_auto_refresh());
    ui_->ContrastMinDoubleSpinBox->setEnabled(!api.contrast.get_contrast_auto_refresh());
    ui_->ContrastMinDoubleSpinBox->setValue(api.contrast.get_contrast_min());
    ui_->ContrastMaxDoubleSpinBox->setEnabled(!api.contrast.get_contrast_auto_refresh());
    ui_->ContrastMaxDoubleSpinBox->setValue(api.contrast.get_contrast_max());

    ui_->actionSettings->setEnabled(api.input.get_camera_kind() != CameraKind::NONE);

    ui_->RecordedEyePushButton->setText(QString::fromStdString(gui::get_recorded_eye_display_string()));
}

void LightUI::set_contrast_mode(bool value) { API.contrast.set_contrast_enabled(value); }

void LightUI::set_contrast_min(const double value) { API.contrast.set_contrast_min(value); }

void LightUI::set_contrast_max(const double value) { API.contrast.set_contrast_max(value); }

void LightUI::set_contrast_auto_refresh(bool value)
{
    API.contrast.set_contrast_auto_refresh(value);
    notify(); // Enable or disable the DoubleBox range
}

void LightUI::change_camera(CameraKind camera) { main_window_->change_camera(camera); }

void LightUI::camera_none() { change_camera(CameraKind::NONE); }

void LightUI::camera_phantom() { change_camera(CameraKind::Phantom); }

void LightUI::camera_ametek_s991_coaxlink_qspf_plus() { change_camera(CameraKind::AmetekS991EuresysCoaxlinkQSFP); }

void LightUI::camera_ametek_s711_coaxlink_qspf_plus() { change_camera(CameraKind::AmetekS711EuresysCoaxlinkQSFP); }

void LightUI::configure_camera()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(API.input.get_camera_ini_name())));
}

void LightUI::set_recordProgressBar_color(const QColor& color, const QString& text)
{
    ui_->recordProgressBar->setStyleSheet("QProgressBar::chunk { background-color: " + color.name() +
                                          "; } "
                                          "QProgressBar { text-align: center; padding-top: 2px; }");
    ui_->recordProgressBar->setFormat(text);
}

void LightUI::pipeline_active(bool active)
{
    ui_->startButton->setEnabled(active);
    ui_->ZSpinBox->setEnabled(active);
    ui_->ZSlider->setEnabled(active);
}

void LightUI::set_window_size_position(int width, int height, int x, int y)
{
    this->resize(width, height);
    this->move(x, y);
}

void LightUI::activate_start_button(bool activate) { ui_->startButton->setEnabled(activate); }

void LightUI::set_progress_bar_value(int value) { ui_->recordProgressBar->setValue(value); }

void LightUI::set_progress_bar_maximum(int maximum) { ui_->recordProgressBar->setMaximum(maximum); }

void LightUI::open_configuration_ui()
{
    main_window_->show();
    main_window_->notify();
    this->hide();
    visible_ = false;
}

void LightUI::set_preset()
{
    std::filesystem::path dest = __PRESET_FOLDER_PATH__ / "preset.json";
    main_window_->reload_ini(dest.string());
    LOG_INFO("Preset loaded");
}

void LightUI::closeEvent(QCloseEvent* event) { main_window_->close(); }

void LightUI::update_recorded_eye()
{
    API.record.set_recorded_eye(API.record.get_recorded_eye() == RecordedEyeType::LEFT ? RecordedEyeType::RIGHT
                                                                                       : RecordedEyeType::LEFT);
    notify();
}

void LightUI::reset_recorded_eye()
{
    API.record.set_recorded_eye(RecordedEyeType::NONE);
    notify();
}

} // namespace holovibes::gui
