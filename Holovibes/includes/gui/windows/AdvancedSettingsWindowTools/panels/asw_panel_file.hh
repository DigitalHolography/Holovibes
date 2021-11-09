#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
class ASWPanelFile : public AdvancedSettingsWindowPanel
{
  public:
    ASWPanelFile(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelFile();

  private:
    QVBoxLayout* file_layout_;
    QPathSelectorLayout* default_input_folder_;
    QPathSelectorLayout* default_output_folder_;
    QPathSelectorLayout* batch_input_folder_;
};
} // namespace holovibes::gui