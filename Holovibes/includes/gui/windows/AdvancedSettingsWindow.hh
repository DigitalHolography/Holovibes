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
    /*! \brief AdvancedSettingsWindow constructor */

    /*! \brief Advanced Settings Window obconstructorject
     *
     * \param parent the object that will embed the layouts
     * \param specific_panel the external panel to plug
     */

    AdvancedSettingsWindow(QMainWindow* parent = nullptr, AdvancedSettingsWindowPanel* specific_panel = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

  private:
    /*! \brief Creates a buffer size panel */
    void create_buffer_size_panel();
    /*! \brief Create a advanced panel */
    void create_advanced_panel();
    /*! \brief Create a file panel */
    void create_file_panel();
    /*! \brief Create a chart panel */
    void create_chart_panel();
    /*! \brief Link/Plug the given panel to the AdvancedSettingWindow (this)
     *
     * \param specific_panel The given panel to plug
     */
    void plug_specific_panel(AdvancedSettingsWindowPanel* specific_panel);

  private:
    QWidget* main_widget_;
    QGridLayout* main_layout_;

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);
};
} // namespace holovibes::gui
