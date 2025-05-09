/*!
 * \file info_panel.cc
 *
 * \brief Definitions for the InfoPanel
 */

#include "API.hh"
#include "info_panel.hh"
#include "MainWindow.hh"
#include "benchmark_worker.hh"
#include "logger.hh"

namespace holovibes::gui
{
InfoPanel::InfoPanel(QWidget* parent)
    : Panel(parent)
    , timer_()
{
    connect(&timer_, SIGNAL(timeout()), this, SLOT(update_information()));
    timer_.start(50);
}

InfoPanel::~InfoPanel() {}

void InfoPanel::init()
{
    set_visible_file_reader_progress(false);
    set_visible_record_progress(false);
}

void InfoPanel::handle_progress_bar(Information& information)
{
    if (!api_.record.is_recording())
        return;

    if (!information.record_info)
    {
        LOG_WARN("Unable to retrieve recording information, the progress bar will not be updated");
        return;
    }
    holovibes::RecordProgressInfo progress = information.record_info.value();

    // When all frames are acquired, we switch the color and the progress bar now tracks the saving progress
    bool saving = !api_.record.get_frame_acquisition_enabled();
    int value = static_cast<int>(progress.acquired_frames);

    if (saving)
    {
        ui_->InfoPanel->set_recordProgressBar_color(QColor(48, 143, 236), "Saving: %v/%m");
        parent_->light_ui_->set_recordProgressBar_color(QColor(48, 143, 236), "Saving...");

        value = static_cast<int>(progress.saved_frames);
    }
    else
        ui_->InfoPanel->set_recordProgressBar_color(QColor(209, 90, 25), "Acquisition: %v/%m");

    ui_->RecordProgressBar->setMaximum(static_cast<int>(progress.total_frames));
    ui_->RecordProgressBar->setValue(value);

    parent_->light_ui_->actualise_record_progress(value, static_cast<int>(progress.total_frames));
}

void InfoPanel::update_information()
{
    ui_->InfoTextEdit->display_information();

    Information information = API.information.get_information();

    handle_progress_bar(information);

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
        ui_->RecordProgressBar->setMaximum(static_cast<int>(max_size));
        ui_->RecordProgressBar->setValue(static_cast<int>(value));

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

void InfoPanel::set_visible_file_reader_progress(bool visible) { ui_->FileReaderProgressBar->setVisible(visible); }

void InfoPanel::set_visible_record_progress(bool visible)
{
    if (visible)
        ui_->RecordProgressBar->reset();

    ui_->RecordProgressBar->setVisible(visible);
}

void InfoPanel::set_recordProgressBar_color(const QColor& color, const QString& text)
{
    ui_->RecordProgressBar->setStyleSheet("QProgressBar::chunk { background-color: " + color.name() +
                                          "; } "
                                          "QProgressBar { text-align: center; padding-top: 2px; }");
    ui_->RecordProgressBar->setFormat(text);
}

} // namespace holovibes::gui
