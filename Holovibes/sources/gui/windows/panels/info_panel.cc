/*!
 * \file info_panel.cc
 *
 * \brief Definitions for the InfoPanel
 */

#include "api.hh"
#include "info_panel.hh"
#include "MainWindow.hh"
#include "benchmark_worker.hh"
#include "logger.hh"

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
    , record_finished_subscriber_("record_finished", [this](bool success) { set_visible_record_progress(false); })
{
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update_information()));
    timer_.start(50);
    chrono_.start();
}

InfoPanel::~InfoPanel() {}

void InfoPanel::init()
{
    set_visible_file_reader_progress(false);
    set_visible_record_progress(false);
}

void InfoPanel::update_information()
{
    chrono_.stop();
    size_t waited_time = chrono_.get_milliseconds();
    if (waited_time >= 1000)
    {
        ui_->InfoTextEdit->display_information_slow(waited_time);
        chrono_.start();
    }
    ui_->InfoTextEdit->display_information();

    Information information;
    API.information.get_information(&information);
    for (auto const& [key, info] : information.progresses)
        update_progress(key, info.current_size, info.max_size);
}

void InfoPanel::update_progress(ProgressType type, const size_t value, const size_t max_size)
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

        NotifierManager::notify<RecordProgressData>(
            "record_progress",
            RecordProgressData{static_cast<int>(value), static_cast<int>(max_size)});
        break;
    default:
        return;
    };
}

void InfoPanel::load_gui(const json& j_us)
{
    bool h = json_get_or_default(j_us, isHidden(), "panels", "info hidden");
    ui_->actionInfo->setChecked(!h);
    setHidden(h);
}

void InfoPanel::save_gui(json& j_us) { j_us["panels"]["info hidden"] = isHidden(); }

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

void InfoPanel::set_recordProgressBar_color(const QColor& color, const QString& text)
{
    ui_->RecordProgressBar->setStyleSheet("QProgressBar::chunk { background-color: " + color.name() +
                                          "; } "
                                          "QProgressBar { text-align: center; padding-top: 2px; }");
    ui_->RecordProgressBar->setFormat(text);
}

} // namespace holovibes::gui
