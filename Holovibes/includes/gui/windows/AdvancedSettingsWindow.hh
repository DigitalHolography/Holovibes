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
 * - emit a signal that can be catched by AdvancedWettingsWindow when
 *    an layout object such as QDoubleSpinBox or QPathSelectorLayout is modified by
 *    the user
 * - comment all .hh files according to Doxygen standard
 * - create a panel class which will organised GroupBoxs
 * - each group box item should be init in a single method, in the associated panel
 * - init every groupbox from holovibes values
 * - override buttons and all Qt object to be more customized
 * - remove
 *   'QLayout: Attempting to add QLayout "" to holovibes::gui::AdvancedSettingsWindow "", which already has a layout'
 *   error
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

  private:
    QGroupBox* create_group_box(const std::string& name);

    QGroupBox* create_buffer_size_group_box(const std::string& name);

    QGroupBox* create_advanced_group_box(const std::string& name);

    QGroupBox* create_file_group_box(const std::string& name);

    QGroupBox* create_chart_group_box(const std::string& name);

  public slots:
    /*! \brief emit signal closed on window is closed */
    void closeEvent(QCloseEvent* event);
};
} // namespace holovibes::gui
