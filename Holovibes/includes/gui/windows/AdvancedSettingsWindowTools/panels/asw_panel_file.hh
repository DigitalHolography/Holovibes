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
    Q_OBJECT

  public:
    ASWPanelFile(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelFile();

  private slots:
    /*! \brief Processing when input folder has changed changes */
    void on_change_input_folder();
    /*! \brief Processing when output folder has changed */
    void on_change_output_folder();
    /*! \brief Processing when batch input folder has changed */
    void on_change_batch_input_folder();

  private:
    QVBoxLayout* file_layout_;
    QPathSelectorLayout* default_input_folder_;
    QPathSelectorLayout* default_output_folder_;
    QPathSelectorLayout* batch_input_folder_;
};
} // namespace holovibes::gui