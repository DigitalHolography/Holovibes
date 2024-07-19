/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Info panel
 */
#pragma once

#include "lightui.hh"
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

    void set_light_ui(std::shared_ptr<LightUI> light_ui);

    void load_gui(const json& j_us) override;
    void save_gui(json& j_us) override;

    /*! \brief Change the text in the text area */
    void set_text(const char* text);

    /*! \brief Show or hide the file reader progress */
    void set_visible_file_reader_progress(bool visible);

    /*! \brief Show or hide the record progress */
    void set_visible_record_progress(bool visible);

    /*! \brief Set the value of the record progress bar */
    void set_recordProgressBar_color(const QColor& color, const QString& text);

  private:
    int height_ = 0;
    int resize_again_ = 0;
    std::shared_ptr<LightUI> light_ui_;
    Subscriber<bool> record_finished_subscriber_;    ///< Subscriber for record finished events.
};
} // namespace holovibes::gui
