#include "info_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
    , parent_(find_main_window(parent))
{
}

InfoPanel::~InfoPanel() {}

void InfoPanel::set_text(const char* text) { parent_->ui.InfoTextEdit->setText(text); }

void InfoPanel::init_file_reader_progress(int value, int max)
{
    parent_->ui.FileReaderProgressBar->setMaximum(static_cast<int>(max));
    parent_->ui.FileReaderProgressBar->setValue(static_cast<int>(value));
}

void InfoPanel::set_visible_file_reader_progress(bool visible)
{
    if (visible)
    {
        parent_->ui.FileReaderProgressBar->show();
    }
    else
    {
        parent_->ui.FileReaderProgressBar->hide();
    }
}

void InfoPanel::update_file_reader_progress(int value) { parent_->ui.FileReaderProgressBar->setValue(value); }

void InfoPanel::init_record_progress(int value, int max)
{
    parent_->ui.RecordProgressBar->setMaximum(static_cast<int>(max));
    parent_->ui.RecordProgressBar->setValue(static_cast<int>(value));
}

void InfoPanel::set_visible_record_progress(bool visible)
{
    if (visible)
    {
        parent_->ui.RecordProgressBar->reset();
        parent_->ui.RecordProgressBar->show();
    }
    else
    {
        parent_->ui.RecordProgressBar->hide();
    }
}
} // namespace holovibes::gui
