/*! \file
 *
 * \brief Regroup all functions used to interact with the composite settings.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

/*! \name Composite
 * \{
 */
inline CompositeKind get_composite_kind() noexcept { return GET_SETTING(CompositeKind); }
inline void set_composite_kind(CompositeKind value) { UPDATE_SETTING(CompositeKind, value); }

inline bool get_composite_auto_weights() noexcept { return GET_SETTING(CompositeAutoWeights); }
inline void set_composite_auto_weights(bool value)
{
    UPDATE_SETTING(CompositeAutoWeights, value);
    pipe_refresh();
}
/*! \} */

/*! \name HSV
 * \{
 */
inline CompositeRGB get_rgb() noexcept { return GET_SETTING(RGB); }
inline void set_rgb(CompositeRGB value) { UPDATE_SETTING(RGB, value); }

inline float get_weight_r() noexcept { return GET_SETTING(RGB).weight.r; }
inline float get_weight_g() noexcept { return GET_SETTING(RGB).weight.g; }
inline float get_weight_b() noexcept { return GET_SETTING(RGB).weight.b; }
inline uint get_composite_p_red() { return GET_SETTING(RGB).frame_index.min; }
inline uint get_composite_p_blue() { return GET_SETTING(RGB).frame_index.max; }

inline void set_weight_r(double value) { SET_SETTING(RGB, weight.r, value); }
inline void set_weight_g(double value) { SET_SETTING(RGB, weight.g, value); }
inline void set_weight_b(double value) { SET_SETTING(RGB, weight.b, value); }
inline void set_weight_rgb(double r, double g, double b)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.weight.r = r;
    rgb.weight.g = g;
    rgb.weight.b = b;
    UPDATE_SETTING(RGB, rgb);
}
inline void set_rgb_p(int min, int max)
{
    holovibes::CompositeRGB rgb = get_rgb();
    rgb.frame_index.min = min;
    rgb.frame_index.max = max;
    UPDATE_SETTING(RGB, rgb);
}
/*! \} */

/*! \name HSV
 * \{
 */
inline CompositeHSV get_hsv() noexcept { return GET_SETTING(HSV); }
inline void set_hsv(CompositeHSV value) { UPDATE_SETTING(HSV, value); }

/*! \name Hue Getters
 * \{
 */
inline uint get_composite_p_min_h() noexcept { return GET_SETTING(HSV).h.frame_index.min; }
inline uint get_composite_p_max_h() noexcept { return GET_SETTING(HSV).h.frame_index.max; }
inline float get_composite_low_h_threshold() noexcept { return GET_SETTING(HSV).h.threshold.min; }
inline float get_composite_high_h_threshold() noexcept { return GET_SETTING(HSV).h.threshold.max; }
inline float get_slider_h_threshold_min() noexcept { return GET_SETTING(HSV).h.slider_threshold.min; }
inline float get_slider_h_threshold_max() noexcept { return GET_SETTING(HSV).h.slider_threshold.max; }
inline float get_slider_h_shift_min() { return GET_SETTING(HSV).h.slider_shift.min; }
inline float get_slider_h_shift_max() { return GET_SETTING(HSV).h.slider_shift.max; }
/*! \} */

/*! \name Hue Setters
 * \{
 */
inline void set_composite_p_min_h(uint value) { SET_SETTING(HSV, h.frame_index.min, value); }
inline void set_composite_p_max_h(uint value) { SET_SETTING(HSV, h.frame_index.max, value); }
inline void set_composite_low_h_threshold(float value) { SET_SETTING(HSV, h.threshold.min, value); }
inline void set_composite_high_h_threshold(float value) { SET_SETTING(HSV, h.threshold.max, value); }
inline void set_slider_h_threshold_min(float value) { SET_SETTING(HSV, h.slider_threshold.min, value); }
inline void set_slider_h_threshold_max(float value) { SET_SETTING(HSV, h.slider_threshold.max, value); }
inline void set_slider_h_shift_min(float value) { SET_SETTING(HSV, h.slider_shift.min, value); }
inline void set_slider_h_shift_max(float value) { SET_SETTING(HSV, h.slider_shift.max, value); }
inline void set_composite_p_h(int min, int max)
{
    holovibes::CompositeHSV hsv = get_hsv();
    hsv.h.frame_index.min = min;
    hsv.h.frame_index.max = max;
    UPDATE_SETTING(HSV, hsv);
}
/*! \} */

/*! \name Saturation Getters
 * \{
 */
inline uint get_composite_p_min_s() noexcept { return GET_SETTING(HSV).s.frame_index.min; }
inline uint get_composite_p_max_s() noexcept { return GET_SETTING(HSV).s.frame_index.max; }
inline float get_composite_low_s_threshold() noexcept { return GET_SETTING(HSV).s.threshold.min; }
inline float get_composite_high_s_threshold() noexcept { return GET_SETTING(HSV).s.threshold.max; }
inline float get_slider_s_threshold_min() noexcept { return GET_SETTING(HSV).s.slider_threshold.min; }
inline float get_slider_s_threshold_max() noexcept { return GET_SETTING(HSV).s.slider_threshold.max; }
inline bool get_composite_p_activated_s() noexcept { return GET_SETTING(HSV).s.frame_index.activated; }
/*! \} */

/*! \name Saturation Setters
 * \{
 */
inline void set_composite_p_min_s(uint value) { SET_SETTING(HSV, s.frame_index.min, value); }
inline void set_composite_p_max_s(uint value) { SET_SETTING(HSV, s.frame_index.max, value); }
inline void set_composite_low_s_threshold(float value) { SET_SETTING(HSV, s.threshold.min, value); }
inline void set_composite_high_s_threshold(float value) { SET_SETTING(HSV, s.threshold.max, value); }
inline void set_slider_s_threshold_min(float value) { SET_SETTING(HSV, s.slider_threshold.min, value); }
inline void set_slider_s_threshold_max(float value) { SET_SETTING(HSV, s.slider_threshold.max, value); }
inline void set_composite_p_activated_s(bool value) { SET_SETTING(HSV, s.frame_index.activated, value); }
/*! \} */

/*! \name Value Getters
 * \{
 */
inline uint get_composite_p_min_v() noexcept { return GET_SETTING(HSV).v.frame_index.min; }
inline uint get_composite_p_max_v() noexcept { return GET_SETTING(HSV).v.frame_index.max; }
inline float get_composite_low_v_threshold() noexcept { return GET_SETTING(HSV).v.threshold.min; }
inline float get_composite_high_v_threshold() noexcept { return GET_SETTING(HSV).v.threshold.max; }
inline float get_slider_v_threshold_min() noexcept { return GET_SETTING(HSV).v.slider_threshold.min; }
inline float get_slider_v_threshold_max() noexcept { return GET_SETTING(HSV).v.slider_threshold.max; }
inline bool get_composite_p_activated_v() noexcept { return GET_SETTING(HSV).v.frame_index.activated; }
/*! \} */

/*! \name Value Setters
 * \{
 */
inline void set_composite_p_min_v(uint value) { SET_SETTING(HSV, v.frame_index.min, value); }
inline void set_composite_p_max_v(uint value) { SET_SETTING(HSV, v.frame_index.max, value); }
inline void set_composite_low_v_threshold(float value) { SET_SETTING(HSV, v.threshold.min, value); }
inline void set_composite_high_v_threshold(float value) { SET_SETTING(HSV, v.threshold.max, value); }
inline void set_slider_v_threshold_min(float value) { SET_SETTING(HSV, v.slider_threshold.min, value); }
inline void set_slider_v_threshold_max(float value) { SET_SETTING(HSV, v.slider_threshold.max, value); }
inline void set_composite_p_activated_v(bool value) { SET_SETTING(HSV, v.frame_index.activated, value); }
/*! \} */
/*! \} */

/*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values
 *
 * \param composite_p_red the new value
 * \param composite_p_blue the new value
 */
void set_composite_intervals(int composite_p_red, int composite_p_blue);

/*! \brief Modifies HSV Hue min frequence
 *
 * \param composite_p_min_h the new value
 */
void set_composite_intervals_hsv_h_min(uint composite_p_min_h);

/*! \brief Modifies HSV Hue max frequence
 *
 * \param composite_p_max_h the new value
 */
void set_composite_intervals_hsv_h_max(uint composite_p_max_h);

/*! \brief Modifies HSV Hue max frequence
 *
 * \param composite_p_min_s the new value
 */
void set_composite_intervals_hsv_s_min(uint composite_p_min_s);

/*! \brief Modifies HSV Saturation max frequence
 *
 * \param composite_p_max_s the new value
 */
void set_composite_intervals_hsv_s_max(uint composite_p_max_s);

/*! \brief Modifies HSV Value min frequence
 *
 * \param composite_p_min_v the new value
 */
void set_composite_intervals_hsv_v_min(uint composite_p_min_v);

/*!
 * \brief Modifies HSV Value min frequence
 *
 * \param composite_p_max_v the new value
 */
void set_composite_intervals_hsv_v_max(uint composite_p_max_v);

/*! \brief Modifies the RGB
 *
 * \param weight_r the new value of Red
 * \param weight_g the new value of Green
 * \param weight_b the new value of blue
 */
void set_composite_weights(double weight_r, double weight_g, double weight_b);

/*! \brief Switchs between to RGB mode
 *
 */
void select_composite_rgb();

/*! \brief Switchs between to HSV mode
 *
 */
void select_composite_hsv();

/*! \brief Enables or disables Saturation frequency channel min and max
 *
 * \param composite_p_activated_s true: enable, false: disable
 */
void actualize_frequency_channel_s(bool composite_p_activated_s);

/*! \brief Enables or disables Value frequency channel min and max
 *
 * \param composite_p_activated_v true: enable, false: disable
 */
void actualize_frequency_channel_v(bool composite_p_activated_v);

} // namespace holovibes::api