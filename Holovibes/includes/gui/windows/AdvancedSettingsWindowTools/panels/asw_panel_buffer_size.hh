#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
class ASWPanelBufferSize : public AdvancedSettingsWindowPanel
{
  public:
    ASWPanelBufferSize(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelBufferSize();

  private:
    QVBoxLayout* buffer_size_layout_;
    QIntSpinBoxLayout* file_;
    QIntSpinBoxLayout* input_;
    QIntSpinBoxLayout* record_;
    QIntSpinBoxLayout* output_;
    QIntSpinBoxLayout* cuts_;
};
} // namespace holovibes::gui