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
/*! \class ASWPanelChart
 *
 * \brief Frame of ASWPanelChart in charge of Chart display settings
 */
class ASWPanelChart : public AdvancedSettingsWindowPanel
{
    Q_OBJECT

  public:
    ASWPanelChart(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr);
    ~ASWPanelChart();

  private:
    /*! \brief Creates attribute auto scale point threshold */
    void create_auto_scale_point_threshold_widget();

  private slots:
    /*! \brief Processing when auto scale point threshold value has changed */
    void on_change_auto_scale_point_threshold_value();

  private:
    QVBoxLayout* chart_layout_;
    QIntSpinBoxLayout* auto_scale_point_threshold_;
};
} // namespace holovibes::gui