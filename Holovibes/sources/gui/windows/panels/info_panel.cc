#include "info_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "global_state_holder.hh"
#include "API.hh"

namespace api = ::holovibes::api;

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
{
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
                case ProgressType::FILE_READ:
                    ui_->FileReaderProgressBar->setMaximum(static_cast<int>(max_size));
                    ui_->FileReaderProgressBar->setValue(static_cast<int>(value));
                    break;
                case ProgressType::CHART_RECORD:
                case ProgressType::FRAME_RECORD:
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

void InfoPanel::load_gui(const boost::property_tree::ptree& ptree)
{
    bool h = ptree.get<bool>("window.info_hidden", isHidden());
    ui_->actionInfo->setChecked(!h);
    setHidden(h);
}

void InfoPanel::save_gui(boost::property_tree::ptree& ptree) { ptree.put<bool>("window.info_hidden", isHidden()); }

void InfoPanel::set_text(const char* text) { ui_->InfoTextEdit->setText(text); }

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
