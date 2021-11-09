#pragma once

/*! \file
 *
 * \brief Qt Advanced settings window class.
 */
#pragma once

#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QGroupBox>

/**
 * TODO: list
 * - add value of changes in the emitted signals
 * - link signals to API
 * - create each 'button' in a single method for each panel
 * - init every groupbox from holovibes values
 * - override buttons and all Qt object to be more customized
 * - remove useless param in ctor (parent)
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
    /*! \brief AdvancedSettingsWindow constructor
     *
     * Create a AdvancedSettingsWindow and shows it.
     *
     */
    AdvancedSettingsWindow(QMainWindow* parent = nullptr);

    /*! \brief Destroy the AdvancedSettingsWindow object. */
    ~AdvancedSettingsWindow();

    QWidget* main_widget;
    QHBoxLayout* main_layout;
    QVBoxLayout* buffer_size_layout;

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);
};
} // namespace holovibes::gui
