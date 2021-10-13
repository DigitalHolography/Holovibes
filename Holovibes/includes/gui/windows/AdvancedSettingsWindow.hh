/*! \file
 *
 * \brief Qt Advanced settings window class.
 */
#pragma once

#include "ui_advancedsettingswindow.h"

/* Forward declarations. */
namespace holovibes
{
}

namespace holovibes::gui
{
/*! \class AdvancedSettingsWindow
 *
 * \brief Qt main window class containing a plot of computed chart values.
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
    AdvancedSettingsWindow(QWidget* parent = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);

  private:
    Ui::AdvancedSettingsWindow ui;
};
} // namespace holovibes::gui
