/*! \file
 *
 */

#include "advanced_settings_window_panel.hh"

namespace holovibes::gui
{
AdvancedSettingsWindowPanel::AdvancedSettingsWindowPanel(const std::string& name)
    : QGroupBox()
    , name_(name)

{
    setTitle(QString::fromUtf8(name.c_str()));
}
AdvancedSettingsWindowPanel::~AdvancedSettingsWindowPanel() {}
} // namespace holovibes::gui