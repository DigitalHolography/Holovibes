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
    /*! \brief Processing when file value has changed */
    void on_change_file_value();
    /*! \brief Processing when input value has changed */
    void on_change_input_value();
    /*! \brief Processing when record value has changed */
    void on_change_record_value();
    /*! \brief Processing when output value has changed */
    void on_change_output_value();
    /*! \brief Processing when cuts value has changed */
    void on_change_cuts_value();

  private:
    QVBoxLayout* buffer_size_layout_;
    QIntSpinBoxLayout* file_;
    QIntSpinBoxLayout* input_;
    QIntSpinBoxLayout* record_;
    QIntSpinBoxLayout* output_;
    QIntSpinBoxLayout* cuts_;
};
} // namespace holovibes::gui