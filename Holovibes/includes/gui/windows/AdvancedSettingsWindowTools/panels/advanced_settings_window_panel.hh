/*! \file
 *
 * \brief Customization of QGroupBox items from Qt.
 */
#pragma once

#include <QGroupBox>
#include "logger.hh"
namespace holovibes::gui
{
/*! \class AdvancedSettingsWindowPanel
 *
 * \brief QGroupBox overload to split AdvancedSettingsWindow in multiple parts
 */
class AdvancedSettingsWindowPanel : public QGroupBox
{
  public:
    /*!
     * \brief Advanced Settings Window Panel object constructor
     *
     * \param parent the object that will embed the layouts
     * \param parent_widget the object that will embed the object
     * \param name the name to display for the created QGroupBox
     */
    AdvancedSettingsWindowPanel(QMainWindow* parent = nullptr,
                                QWidget* parent_widget = nullptr,
                                const std::string& name = "");

    /*! \brief Advanced Settings Window Panel object desctructor */
    ~AdvancedSettingsWindowPanel();

  public:
    QMainWindow* parent_;
    QWidget* parent_widget_;
    const std::string& name_;
};
} // namespace holovibes::gui