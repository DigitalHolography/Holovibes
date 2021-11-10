#include "advanced_settings_window_panel.hh"

namespace holovibes::gui
{
AdvancedSettingsWindowPanel::AdvancedSettingsWindowPanel(QMainWindow* parent,
                                                         QWidget* parent_widget,
                                                         const std::string& name)
    : QGroupBox(parent)
    , parent_widget_(parent_widget)
    , name_(name)

{
    setTitle(QString::fromUtf8(name.c_str()));
}
AdvancedSettingsWindowPanel::~AdvancedSettingsWindowPanel() {}
} // namespace holovibes::gui