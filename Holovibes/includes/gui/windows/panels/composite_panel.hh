/*! \file
 *
 * \brief File containing methods, attributes and slots related to the Composite panel
 */
#pragma once

#include "panel.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class CompositePanel
 *
 * \brief Class representing the Composite Panel in the GUI
 */
class CompositePanel : public Panel
{
    Q_OBJECT

  public:
    CompositePanel(QWidget* parent = nullptr);
    ~CompositePanel();

    void on_notify() override;

  public slots:

    /*! \brief Enable or disable the FFT shift on Z axis */
    void click_z_fft_shift(bool checked);

    /*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values */
    void set_composite_intervals();

    /*! \brief Modifies HSV Hue min frequence */
    void set_composite_intervals_hsv_h_min();

    /*! \brief Modifies HSV Hue max frequence*/
    void set_composite_intervals_hsv_h_max();

    /*! \brief Modifies HSV Saturation min frequence */
    void set_composite_intervals_hsv_s_min();

    /*! \brief Modifies HSV Saturation max frequence */
    void set_composite_intervals_hsv_s_max();

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_min();

    /*! \brief Modifies HSV Value min frequence */
    void set_composite_intervals_hsv_v_max();

    /*! \brief Modifies the RGV from ui values */
    void set_composite_weights();

    /*! \brief Automatic equalization (Auto-constrast)
     *
     * \param value true: enable, false: disable
     */
    void set_composite_auto_weights(bool value);

    /*! \brief Switchs between RGB mode and HSV mode */
    void click_composite_rgb_or_hsv();

    /*! \brief Modifies Hue min threshold and guaratees that Hue min threshold does not exceed Hue max threshold */
    void slide_update_threshold_h_min();

    /*! \brief Modifies Hue max threshold and guaratees that Hue max threshold is higher than Hue min threshold */
    void slide_update_threshold_h_max();

    void slide_update_shift_h_min();
    void slide_update_shift_h_max();

    /*! \brief Change Saturation min threshold. Saturation min threshold does not exceed max threshold */
    void slide_update_threshold_s_min();

    /*! \brief Change Saturation max. Saturation max threshold is higher than min threshold */
    void slide_update_threshold_s_max();

    /*! \brief Change Value min threshold and guarantee that Value min threshold does not exceed Value max threshold */
    void slide_update_threshold_v_min();

    /*! \brief Modify Value max threshold and guarantee that Value max threshold is higher than Value min threshold */
    void slide_update_threshold_v_max();

    /*! \brief Enables or disables Saturation frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_s();

    /*! \brief Enables or disables Value frequency channel min and max from ui checkbox */
    void actualize_frequency_channel_v();

    /*! \brief Make the ui composite overlay visible */
    void set_composite_area();
};
} // namespace holovibes::gui
