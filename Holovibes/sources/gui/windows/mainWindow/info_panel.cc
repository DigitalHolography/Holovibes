#include "info_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
{
}

InfoPanel::~InfoPanel() {}

void InfoPanel::on_notify() {}

void InfoPanel::set_text(const char* text) { ui_->InfoTextEdit->setText(text); }

void InfoPanel::init_file_reader_progress(int value, int max)
{
    ui_->FileReaderProgressBar->setMaximum(static_cast<int>(max));
    ui_->FileReaderProgressBar->setValue(static_cast<int>(value));
}

void InfoPanel::set_visible_file_reader_progress(bool visible)
{
    if (visible)
    {
        ui_->FileReaderProgressBar->show();
    }
    else
    {
        ui_->FileReaderProgressBar->hide();
    }
}

void InfoPanel::update_file_reader_progress(int value) { ui_->FileReaderProgressBar->setValue(value); }

void InfoPanel::init_record_progress(int value, int max)
{
    ui_->RecordProgressBar->setMaximum(static_cast<int>(max));
    ui_->RecordProgressBar->setValue(static_cast<int>(value));
}

void InfoPanel::set_visible_record_progress(bool visible)
{
    if (visible)
    {
        ui_->RecordProgressBar->reset();
        ui_->RecordProgressBar->show();
    }
    else
    {
        ui_->RecordProgressBar->hide();
    }
}
} // namespace holovibes::gui
