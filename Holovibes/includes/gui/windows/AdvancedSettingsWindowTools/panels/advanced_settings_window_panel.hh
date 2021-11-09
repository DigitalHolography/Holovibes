#pragma once

#include <QGroupBox>

namespace holovibes::gui
{
class AdvancedSettingsWindowPanel : public QGroupBox
{
  protected:
    AdvancedSettingsWindowPanel(QMainWindow* parent = nullptr,
                                QWidget* parent_widget = nullptr,
                                const std::string& name = "");
    ~AdvancedSettingsWindowPanel();

  private:
    QMainWindow* parent_;
    QWidget* parent_widget_;
    const std::string& name_;
};
} // namespace holovibes::gui