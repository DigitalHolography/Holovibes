/*! \file
 *
 * \brief Specialization of AdvancedSettingsWindowPanel class
 */
#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include "QPathSelectorLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
/*! \class ASWPanelFile
 *
 * \brief Frame of ASWPanelFile in charge of File settings from holovibes
 */
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