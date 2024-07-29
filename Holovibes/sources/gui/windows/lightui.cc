#include <filesystem>
#include <cmath> // Pour std::round

#include "lightui.hh"
#include "MainWindow.hh"
#include "export_panel.hh"
#include "image_rendering_panel.hh"
#include "logger.hh"
#include "tools.hh"
#include "API.hh"
#pragma warning(push, 0)
#include "ui_lightui.h"
#pragma warning(pop)

#include "notifier.hh"

namespace holovibes::gui
{
LightUI::LightUI(QWidget* parent, MainWindow* main_window, ExportPanel* export_panel)
    : QMainWindow(parent)
    , ui_(new Ui::LightUI)
    , main_window_(main_window)
    , export_panel_(export_panel)
    , visible_(false)
    , z_distance_subscriber_("z_distance", std::bind(&LightUI::actualise_z_distance, this, std::placeholders::_1))
    , record_start_subscriber_("record_start", std::bind(&LightUI::on_record_start, this, std::placeholders::_1))
    , record_end_subscriber_("record_stop", std::bind(&LightUI::on_record_stop, this, std::placeholders::_1))
    , record_finished_subscriber_("record_finished", [this](bool success)
                                       {
                                           reset_start_button();
                                           reset_record_progress_bar();
                                       })
{
    ui_->setupUi(this);

    connect(ui_->OutputFileBrowseToolButton, &QPushButton::clicked, this, &LightUI::browse_record_output_file_ui);
    connect(ui_->OutputFileNameLineEdit, &QLineEdit::textChanged, this, &LightUI::set_record_file_name);
    connect(ui_->startButton, &QPushButton::toggled, this, &LightUI::start_stop_recording);
    connect(ui_->actionConfiguration_UI, &QAction::triggered, this, &LightUI::open_configuration_ui);
    connect(ui_->actionPreset, &QAction::triggered, this, &LightUI::set_preset);
    connect(ui_->ZSpinBox, &QSpinBox::valueChanged, this, &LightUI::z_value_changed_spinBox);
    connect(ui_->ZSlider, &QSlider::valueChanged, this, &LightUI::z_value_changed_slider);

    ui_->startButton->setStyleSheet("background-color: rgb(50, 50, 50);");
    // ui_->startButton->setShortcut(Qt::CTRL + Qt::Key_R);  #FIXME: This shortcut is not working, even though it works for MainWindow
}

LightUI::~LightUI()
{
    api::write_ui_mode(visible_);

    delete ui_;
}

void LightUI::showEvent(QShowEvent* event)
{
    QMainWindow::showEvent(event);
    visible_ = true;
}

void LightUI::actualise_record_output_file_ui(const std::filesystem::path file_path)
{
    ui_->OutputFilePathLineEdit->setText(QString::fromStdString(file_path.parent_path().string()));
    // remove the extension from the filename
    ui_->OutputFileNameLineEdit->setText(QString::fromStdString(file_path.stem().string()));
}

void LightUI::actualise_z_distance(const double z_distance)
{
    const QSignalBlocker blocker(ui_->ZSpinBox);
    const QSignalBlocker blocker2(ui_->ZSlider);
    ui_->ZSpinBox->setValue(static_cast<int>(std::round(z_distance * 1000)));
    ui_->ZSlider->setValue(static_cast<int>(std::round(z_distance * 1000)));
}

void LightUI::z_value_changed_spinBox(int z_distance)
{
    api::set_z_distance(static_cast<double>(z_distance) / 1000.0f);
}

void LightUI::z_value_changed_slider(int z_distance) { api::set_z_distance(static_cast<double>(z_distance) / 1000.0f); }

void LightUI::browse_record_output_file_ui()
{
    //! FIXME: This is a kind of hack that works with the current implementation of MainWindow. Ideally, lightui should
    //! not know about the MainWindow and the ExportPanel. It should only know about the API. One way to fix it is to
    //! create a new browser in this class and then use notify to send the file path to the API (and synchronize the API
    //! with the file path).
    std::filesystem::path file_path(export_panel_->browse_record_output_file().toStdString());
    std::string file_path_str = file_path.string();
    std::replace(file_path_str.begin(), file_path_str.end(), '/', '\\');
    ui_->OutputFilePathLineEdit->setText(QString::fromStdString(file_path_str));

    // remove the extension from the filename
    ui_->OutputFileNameLineEdit->setText(QString::fromStdString(file_path.stem().string()));
}
void LightUI::set_record_file_name(const QString& filename)
{
    // concatenate the path with the filename
    std::filesystem::path path(ui_->OutputFilePathLineEdit->text().toStdString());
    std::filesystem::path file(filename.toStdString());
    export_panel_->set_output_file_name((path / file).string());
}

void LightUI::start_stop_recording(bool start)
{
    if (start)
    {
        export_panel_->start_record();
    }
    else
    {
        api::stop_record();
    }
}

void LightUI::on_record_start(RecordMode record)
{
    ui_->startButton->setText("Stop recording");
    ui_->startButton->setStyleSheet("background-color: rgb(0, 0, 255);");
    LOG_INFO("Recording started");
}

void LightUI::on_record_stop(RecordMode record)
{
    ui_->startButton->setText("Start recording");
    ui_->startButton->setStyleSheet("background-color: rgb(50, 50, 50);");
    LOG_INFO("Recording stopped");
}

void LightUI::reset_start_button()
{
    ui_->startButton->setChecked(false);
    ui_->startButton->setText("Start recording");
    ui_->startButton->setStyleSheet("background-color: rgb(50, 50, 50);");
}

void LightUI::actualise_record_progress(const int value, const int max)
{
    ui_->recordProgressBar->setMaximum(max);
    ui_->recordProgressBar->setValue(value);
}

void LightUI::reset_record_progress_bar()
{
    set_recordProgressBar_color(QColor(10, 10, 10), "Idle");
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

} // namespace holovibes::gui