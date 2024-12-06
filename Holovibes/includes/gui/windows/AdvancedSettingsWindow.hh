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
     * \param[in] parent the object that will embed the layouts
     */

    AdvancedSettingsWindow(QMainWindow* parent = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

    /*! \brief Set the callback function called when user click on the Save button
     *
     * \param[in] callback the function to call
     */
    void set_callback(std::function<void()> callback) { callback_ = callback; }

  private:
    /*!
     * \brief Change the correspondant folder lineEdit
     *
     * \param[in] lineEdit The line that is currently changed
     */
    void change_folder(Drag_drop_lineedit* lineEdit);

    /*! \brief set ui values according to cd */
    void set_current_values();

  private:
    Ui::AdvancedSettingsWindow ui;

    /*! \brief Callback function called when user click on the Save button */
    std::function<void()> callback_ = []() {};

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);

    /*! \brief set cd_ and mainwindow values according to ui */
    void set_ui_values();

    void change_input_folder_path();
    void change_output_folder_path();
};
} // namespace holovibes::gui
