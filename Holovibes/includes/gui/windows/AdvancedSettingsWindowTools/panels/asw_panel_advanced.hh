#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
class ASWPanelAdvanced : public AdvancedSettingsWindowPanel
{
  public:
    ASWPanelAdvanced(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelAdvanced();

  private:
    QVBoxLayout* advanced_layout_;
    QDoubleSpinBoxLayout* display_rate_;
    QDoubleSpinBoxLayout* filter2d_smooth_low_;
    QDoubleSpinBoxLayout* filter2d_smooth_high_;
    QDoubleSpinBoxLayout* contrast_upper_threshold_;
    QIntSpinBoxLayout* renorm_constant_;
    QIntSpinBoxLayout* cuts_contrast_p_offset_;
};
} // namespace holovibes::gui