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
    /*!
     * \brief Enables or disables Artery mask
     *
     */
    void set_artery_mask(bool enabled);
    /*!
     * \brief Enables or disables Vein mask
     *
     */
    void set_vein_mask(bool enabled);
    /*!
     * \brief Enables or disables Vein mask
     *
     */
    void set_choroid_mask(bool enabled);

    /*! \brief Change the value for the spin box of time window */
    void update_time_window();

    /*! \brief Change the vesselness simga value through the spinbox */
    void update_vesselness_sigma(double value);

    /*! \brief Change the vesselness sigma value through slider, divide the integer [10, 500] value by 100 to normalize
     * in range [0.01, 5] */
    void update_vesselness_sigma_slider(int value);

    void update_min_mask_area(int value);
    void update_min_mask_area_slider(int value);

    void update_diaphragm_factor(double value);
    void update_diaphragm_factor_slider(int value);
    void update_diaphragm_preview(bool enabled);

    void update_barycenter_factor(double value);
    void update_barycenter_factor_slider(int value);
    void update_barycenter_preview(bool enabled);
    // private:
};
} // namespace holovibes::gui
