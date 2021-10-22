/*! \file
 *
 * \brief Qt Advanced settings window class.
 */
#pragma once

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
    /*! \brief AdvancedSettingsWindow constructor
     *
     * Create a AdvancedSettingsWindow and shows it.
     *
     */
    AdvancedSettingsWindow(QMainWindow* parent = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);

    /*! \brief set cd_ and mainwindow values according to ui */
    void set_ui_values();

    void change_input_folder_path();
    void change_output_folder_path();
    void change_batch_input_folder_path();

  private:
    Ui::AdvancedSettingsWindow ui;

    /*!
     * \brief Change the correspondant folder lineEdit
     *
     * \param lineEdit The line that is currently changed
     */
    void change_folder(Drag_drop_lineedit* lineEdit);

    /*! \brief set ui values according to cd */
    void set_current_values();
};
} // namespace holovibes::gui
