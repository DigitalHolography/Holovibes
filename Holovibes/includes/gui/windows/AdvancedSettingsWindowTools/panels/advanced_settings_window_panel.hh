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
     * \param name the name to display for the created QGroupBox
     */
    AdvancedSettingsWindowPanel(const std::string& name = "");

    /*! \brief Advanced Settings Window Panel object desctructor */
    ~AdvancedSettingsWindowPanel();

    virtual void set_ui_values() = 0;

    virtual void set_current_values() = 0;

  public:
    const std::string& name_;
};
} // namespace holovibes::gui