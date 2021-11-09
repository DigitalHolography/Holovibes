/*! \file
 *
 * \brief Specialization of AdvancedSettingsWindowPanel class
 */
#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleSpinBoxLayout.hh"
#include <QVBoxLayout>

namespace holovibes::gui
{
/*! \class ASWPanelBufferSize
 *
 * \brief Frame of ASWPanelBufferSize in charge of Buffers settings from holovibes
 */
class ASWPanelBufferSize : public AdvancedSettingsWindowPanel
{
    Q_OBJECT

  public:
    ASWPanelBufferSize(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelBufferSize();

  private slots:
    void on_change_file_value();

  private:
    QVBoxLayout* buffer_size_layout_;
    QIntSpinBoxLayout* file_;
    QIntSpinBoxLayout* input_;
    QIntSpinBoxLayout* record_;
    QIntSpinBoxLayout* output_;
    QIntSpinBoxLayout* cuts_;
};
} // namespace holovibes::gui