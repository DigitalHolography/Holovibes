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
    /*! \brief Change the value for the spin box of time window */
    void update_time_window();

    /*! \brief Change the vesselness simga value through the spinbox */
    void update_vesselness_sigma(double value);

    /*! \brief Change the vesselness sigma value through slider, divide the integer [10, 500] value by 100 to normalize
     * in range [0.01, 5] */
    void update_vesselness_sigma_slider(int value);

    void set_otsu_kind(int index);
    void set_otsu_window_size(int value);
    void set_otsu_local_threshold(double value);

    // private:
};
} // namespace holovibes::gui
