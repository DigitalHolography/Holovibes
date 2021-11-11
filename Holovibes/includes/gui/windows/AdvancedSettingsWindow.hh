#pragma once

/*! \file
 *
 * \brief Qt Advanced settings window class.
 */
#pragma once

#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QSpinBox>
#include <QGroupBox>

#include "advanced_settings_window_panel.hh"

/**
 * TODO: list
 * - (maybe) save into .ini file every managed params
 * - put each panel in a global to improve organization
 */

namespace holovibes::gui
{
/*! \class AdvancedSettingsWindow
 *
 * \brief Qt main window class allowing to change Advanced Settings.
 */
class AdvancedSettingsWindow : public QMainWindow
{
    Q_OBJECT

  signals:
    void closed();

  public:
    // TODO: refresh comment
    /*! \brief AdvancedSettingsWindow constructor
     *
     * Create a AdvancedSettingsWindow and shows it.
     *
     */
    AdvancedSettingsWindow(QMainWindow* parent = nullptr, AdvancedSettingsWindowPanel* specific_panel = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

  private:
    // TODO: comments
    void create_buffer_size_panel();
    void create_advanced_panel();
    void create_file_panel();
    void create_chart_panel();
    void plug_specific_panel(AdvancedSettingsWindowPanel* specific_panel);

  private:
    QWidget* main_widget_;
    QGridLayout* main_layout_;

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);
};
} // namespace holovibes::gui
