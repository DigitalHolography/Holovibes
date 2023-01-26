/*! \file
 *
 */

#include "info_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "global_state_holder.hh"
#include "API.hh"
#include "user_interface.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().info_panel = this;
}

InfoPanel::~InfoPanel() {}

void InfoPanel::init()
{
    ::holovibes::worker::InformationWorker::update_progress_function_ =
        [=](ProgressType type, const size_t value, const size_t max_size)
    {
        parent_->synchronize_thread(
            [=]()
            {
                switch (type)
                {
                case ProgressType::READ:
                    ui_->FileReaderProgressBar->setMaximum(static_cast<int>(max_size));
                    ui_->FileReaderProgressBar->setValue(static_cast<int>(value));
                    break;
                case ProgressType::RECORD:
                    ui_->RecordProgressBar->setMaximum(static_cast<int>(max_size));
                    ui_->RecordProgressBar->setValue(static_cast<int>(value));
                    break;
                default:
                    return;
                };
            });
    };
    set_visible_file_reader_progress(false);
    set_visible_record_progress(false);
}

void InfoPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, isHidden(), "panels", "info hidden");
    ui_->actionInfo->setChecked(!h);
    setHidden(h);
}

void InfoPanel::save_gui(json& j_us) { j_us["panels"]["info hidden"] = isHidden(); }

void InfoPanel::set_text(const char* text)
{
    QTextEdit* text_edit = ui_->InfoTextEdit;

    text_edit->setText(text);

    // For some reason, the GUI needs multiple updates to return to its base layout
    if (resize_again_-- > 0)
        parent_->adjustSize();

    if (text_edit->document()->size().height() != height_)
    {
        height_ = text_edit->document()->size().height();
        text_edit->setMinimumSize(0, height_);
        parent_->adjustSize();
        resize_again_ = 3;
    }
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