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

void InfoPanel::init()
{
    auto update_progress = [=](InformationContainer::ProgressType type, const size_t value, const size_t max_size) {
        parent_->synchronize_thread([=]() {
            switch (type)
            {
            case InformationContainer::ProgressType::FILE_READ:
                ui_->FileReaderProgressBar->setMaximum(static_cast<int>(max_size));
                ui_->FileReaderProgressBar->setValue(static_cast<int>(value));
                break;
            case InformationContainer::ProgressType::CHART_RECORD:
            case InformationContainer::ProgressType::FRAME_RECORD:
                ui_->RecordProgressBar->setMaximum(static_cast<int>(max_size));
                ui_->RecordProgressBar->setValue(static_cast<int>(value));
                break;
            default:
                return;
            };
        });
    };
    Holovibes::instance().get_info_container().set_update_progress_function(update_progress);
    set_visible_file_reader_progress(false);
    set_visible_record_progress(false);
}

void InfoPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    ui_->actionInfo->setChecked(!ptree.get<bool>("info.hidden", isHidden()));
}

void InfoPanel::save_ini(boost::property_tree::ptree& ptree) { ptree.put<bool>("info.hidden", isHidden()); }

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
