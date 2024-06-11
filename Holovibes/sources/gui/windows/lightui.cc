#include <filesystem>

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
LightUI::LightUI(QWidget* parent,
                 MainWindow* main_window,
                 ExportPanel* export_panel)
    : QMainWindow(parent)
    , ui_(new Ui::LightUI)
    , main_window_(main_window)
    , export_panel_(export_panel)
    , visible_(false)
    , z_distance_subscriber_("z_distance", 
        std::bind(&LightUI::actualise_z_distance, this, std::placeholders::_1)
    )
    , record_start_subscriber_("record_start", 
        std::bind(&LightUI::on_record_start, this, std::placeholders::_1)
    )
    , record_end_subscriber_("record_stop", 
        std::bind(&LightUI::on_record_stop, this, std::placeholders::_1)
    )
{
    ui_->setupUi(this);

    connect(ui_->OutputFileBrowseToolButton, &QPushButton::clicked, this, &LightUI::browse_record_output_file_ui);
    connect(ui_->startButton, &QPushButton::toggled, this, &LightUI::start_stop_recording);
    connect(ui_->actionConfiguration_UI, &QAction::triggered, this, &LightUI::open_configuration_ui);
    connect(ui_->ZSpinBox, &QSpinBox::valueChanged, this, &LightUI::z_value_changed_spinBox);
    connect(ui_->ZSlider, &QSlider::valueChanged, this, &LightUI::z_value_changed_slider);

    actualise_z_distance(api::get_z_distance());

    ui_->recordProgressBar->hide();
    ui_->startButton->setStyleSheet("background-color: rgb(50, 50, 50);");
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

void LightUI::actualise_record_output_file_ui(const QString& filename)
{
    ui_->OutputFilePathLineEdit->setText(filename);
}

void LightUI::actualise_z_distance(const double z_distance)
{
    const QSignalBlocker blocker(ui_->ZSpinBox);
    const QSignalBlocker blocker2(ui_->ZSlider);
    ui_->ZSpinBox->setValue(static_cast<int>(z_distance * 1000));
    ui_->ZSlider->setValue(static_cast<int>(z_distance * 1000));
}

void LightUI::z_value_changed_spinBox(int z_distance)
{
    api::set_z_distance(static_cast<double>(z_distance) / 1000.0f);
}

void LightUI::z_value_changed_slider(int z_distance)
{
    api::set_z_distance(static_cast<double>(z_distance) / 1000.0f);
}

void LightUI::browse_record_output_file_ui()
{
    LOG_INFO("Browsing record output file");
    ui_->OutputFilePathLineEdit->setText(export_panel_->browse_record_output_file());
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
    ui_->startButton->setText("Stop");
    ui_->startButton->setStyleSheet("background-color: rgb(0, 0, 255);");
    LOG_INFO("Recording started");
}

void LightUI::on_record_stop(RecordMode record)
{
    ui_->startButton->setText("Start");
    ui_->startButton->setStyleSheet("background-color: rgb(50, 50, 50);");
    LOG_INFO("Recording stopped");
}

void LightUI::open_configuration_ui()
{
    LOG_INFO("Opening configuration UI");
    main_window_->show();
    this->hide();
    visible_ = false;
}

} // namespace holovibes::gui