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
/*! \class AdvancedSettingsWindowPanel
 *
 * \brief Frame of AdvancedSettingsWindow in charge of Advanced settings from holovibes
 */
class ASWPanelAdvanced : public AdvancedSettingsWindowPanel
{
    Q_OBJECT

  public:
    ASWPanelAdvanced(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelAdvanced();

  private:
    /*! \brief Creates attribute display rate */
    void create_display_rate_widget();
    /*! \brief Creates attribute filter2d smooth low */
    void create_filter2d_smooth_low_widget();
    /*! \brief Creates attribute filter2d smooth high */
    void create_filter2d_smooth_high_widget();
    /*! \brief Creates attribute contrast upper threshold */
    void create_contrast_upper_threshold_widget();
    /*! \brief Creates attribute renorm constant */
    void create_renorm_constant_widget();
    /*! \brief Creates attribute cuts contrast p offset */
    void create_cuts_contrast_p_offset_widget();

  private slots:
    /*! \brief Processing when display rate value has changed */
    void on_change_display_rate_value();
    /*! \brief Processing when filter2d smooth low value has changed */
    void on_change_filter2d_smooth_low_value();
    /*! \brief Processing when filter2d smooth high value has changed */
    void on_change_filter2d_smooth_high_value();
    /*! \brief Processing when upper threshold value has changed */
    void on_change_contrast_upper_threshold_value();
    /*! \brief Processing when renorm constant value has changed */
    void on_change_renorm_constant_value();
    /*! \brief Processing when cuts contrast p offset value has changed */
    void on_change_cuts_contrast_p_offset_value();

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