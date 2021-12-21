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
#include "ui_advancedsettingswindow.h"

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

    /*! \brief Advanced Settings Window
     *
     * \param parent the object that will embed the layouts
     * \param specific_panel the external panel to plug
     */

    AdvancedSettingsWindow(QMainWindow* parent = nullptr, AdvancedSettingsWindowPanel* specific_panel = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

  private:
    /*! \brief Link/Plug the given panel to the AdvancedSettingWindow (this)
     *
     * \param specific_panel The given panel to plug
     */
    void plug_specific_panel(AdvancedSettingsWindowPanel* specific_panel);

    /*!
     * \brief Change the correspondant folder lineEdit
     *
     * \param lineEdit The line that is currently changed
     */
    void change_folder(Drag_drop_lineedit* lineEdit);

    /*! \brief set ui values according to cd */
    void set_current_values();

  private:
    Ui::AdvancedSettingsWindow ui;
    AdvancedSettingsWindowPanel* specific_panel_;

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);

    /*! \brief set cd_ and mainwindow values according to ui */
    void set_ui_values();

    void change_input_folder_path();
    void change_output_folder_path();
    void change_batch_input_folder_path();
};
} // namespace holovibes::gui
