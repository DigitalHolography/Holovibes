/*! \file
 *
 * \brief Specialization of AdvancedSettingsWindowPanel class for MainWindow
 */
#pragma once

#include "advanced_settings_window_panel.hh"
#include "QIntSpinBoxLayout.hh"
#include "QDoubleRefSpinBoxLayout.hh"
#include <QVBoxLayout>

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
    ASWMainWindowPanel(QMainWindow* parent = nullptr, QWidget* parent_widget = nullptr, double* z_step = nullptr);
    ~ASWMainWindowPanel();

  private:
    /*! \brief Creates attribute z_step widget */
    void create_z_step_widget(double* z_step = nullptr);

  private:
    QVBoxLayout* chart_layout_;
    QDoubleRefSpinBoxLayout* z_step_;
};
} // namespace holovibes::gui