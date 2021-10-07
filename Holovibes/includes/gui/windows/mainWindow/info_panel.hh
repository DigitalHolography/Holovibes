/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Info panel
 */
#pragma once

#include "panel.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class InfotPanel
 *
 * \brief Class representing the Info Panel in the GUI
 */
class InfoPanel : public Panel
{
    Q_OBJECT

  public:
    InfoPanel(QWidget* parent = nullptr);
    ~InfoPanel();

    void set_text(const char* text);
    void init_file_reader_progress(int value, int max);
    void init_record_progress(int value, int max);
    void set_visible_file_reader_progress(bool visible);
    void set_visible_record_progress(bool visible);
    void update_file_reader_progress(int value);

  private:
    MainWindow* parent_;
};
} // namespace holovibes::gui
