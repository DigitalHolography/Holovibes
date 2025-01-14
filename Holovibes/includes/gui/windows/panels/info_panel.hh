/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Info panel
 */
#pragma once

#include <QTimer>

#include "gui_info_text_edit.hh"
#include "notifier.hh"
#include "panel.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class InfoPanel
 *
 * \brief Class representing the Info Panel in the GUI
 */
class InfoPanel : public Panel
{
    Q_OBJECT

  public:
    InfoPanel(QWidget* parent = nullptr);
    ~InfoPanel();

    void init() override;

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    /*!
     * \brief Updates the progress bar depending of the progress type
     *
     * \param type The progress type to set in the bar
     * \param value The current state of the bar
     * \param max_size The maximum boundary of the bar
     */
    void update_progress(ProgressType type, const size_t value, const size_t max_size);

    /*! \brief Show or hide the file reader progress */
    void set_visible_file_reader_progress(bool visible);

    /*! \brief Show or hide the record progress */
    void set_visible_record_progress(bool visible);

    /*! \brief Set the value of the record progress bar */
    void set_recordProgressBar_color(const QColor& color, const QString& text);

  public slots:
    /*!
     * \brief Is triggered every 50ms to update the information text
     *
     */
    void update_information();

  private:
    QTimer timer_;

  private:
    void handle_progress_bar(Information& information);
};
} // namespace holovibes::gui
