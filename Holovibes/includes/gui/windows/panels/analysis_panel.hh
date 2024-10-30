/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Analysis panel
 */
#pragma once

#include "enum_record_mode.hh"
#include "panel.hh"
#include "PlotWindow.hh"
#include "lightui.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class AnalysisPanel
 *
 * \brief Class representing the Analysis panel in the GUI
 * This panel contains all analysis algorithms and tools such as mask computing using moments, chart displaying...
 */
class AnalysisPanel : public Panel
{
    Q_OBJECT

  public:
    AnalysisPanel(QWidget* parent = nullptr);
    ~AnalysisPanel();

    void init() override;
    void on_notify() override;

  public slots:
    void set_otsu_kind(int index);
    void set_otsu_window_size(int value);
    void set_otsu_local_threshold(double value);

    void set_bw_area_filter(bool enabled);
    void set_bw_area_filter_value(int value);

    // private:
};
} // namespace holovibes::gui
