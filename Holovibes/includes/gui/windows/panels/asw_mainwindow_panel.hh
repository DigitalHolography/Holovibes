/*! \file
 *
 * \brief Specialization of AdvancedSettingsWindowPanel class for MainWindow
 */
#pragma once

#include "advanced_settings_window_panel.hh"
#include "q_double_spin_box_layout.hh"
#include "image_rendering_panel.hh"

namespace holovibes::gui
{
/*! \class ASWMainWindowPanel
 *
 * \brief Frame of ASWMainWindowPanel in charge of Chart display settings
 */
class ASWMainWindowPanel : public AdvancedSettingsWindowPanel
{
    Q_OBJECT

  public:
    ASWMainWindowPanel(ImageRenderingPanel* parent);
    ~ASWMainWindowPanel();
    void set_ui_values() override;
    void set_current_values() override;

  private:
    /*! \brief Creates attribute z_step widget */
    void create_z_step_widget(QVBoxLayout* layout);

  private:
    // parent_ is only use to access getters and setters
    ImageRenderingPanel* parent_;
    QDoubleSpinBoxLayout* z_step_;
};
} // namespace holovibes::gui