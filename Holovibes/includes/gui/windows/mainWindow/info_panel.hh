/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Info panel
 */
#pragma once

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

    void on_notify() override;

    /*! \brief Change the text in the text area */
    void set_text(const char* text);

    /*! \brief Initialize the file reader progress with base and max values */
    void init_file_reader_progress(int value, int max);
    /*! \brief Show or hdie the file reader progress */
    void set_visible_file_reader_progress(bool visible);
    /*! \brief Update the value of the file readre progress bar */
    void update_file_reader_progress(int value);

    /*! \brief Initialize the record progress with base and max values */
    void init_record_progress(int value, int max);
    /*! \brief Show or hdie the record progress */
    void set_visible_record_progress(bool visible);
};
} // namespace holovibes::gui
