/*! \file
 *
 * \brief Regroup all functions used to interact with the composite settings.
 */
#pragma once

#include "common_api.hh"

// TODO: compute.treshold used in hsv.cu but never modified (not in the GUI)
// Most of pipe refresh used only to redo contrast but just clearing the acc queue is sufficient

namespace holovibes::api
{

/*! \brief Return the composite kind
 *
 * \return Either RGB or HSV
 */
inline CompositeKind get_composite_kind() noexcept { return GET_SETTING(CompositeKind); }

/*! \brief Set the composite kind
 *
 * \param[in] value The new composite kind
 */
inline void set_composite_kind(CompositeKind value) { UPDATE_SETTING(CompositeKind, value); }

inline bool get_z_fft_shift() noexcept { return GET_SETTING(ZFFTShift); }

inline void set_z_fft_shift(bool checked) { UPDATE_SETTING(ZFFTShift, checked); }

#pragma region RGB

/*! \name RGB
 * \{
 */

/*! \brief Return whether the RGB weights are automatically computed or not (similar to auto contrast).
 *
 * \return true if the composite auto weights is enabled
 */
inline bool get_composite_auto_weights() noexcept { return GET_SETTING(CompositeAutoWeights); }

/*! \brief Set whether the RGB weights are automatically computed or not (similar to auto contrast).
 *
 * \param[in] value enable or disable the composite auto weights
 */
inline void set_composite_auto_weights(bool value)
{
    UPDATE_SETTING(CompositeAutoWeights, value);
    pipe_refresh();
}

inline float get_weight_r() noexcept { return GET_SETTING(RGB).weight.r; }
inline float get_weight_g() noexcept { return GET_SETTING(RGB).weight.g; }
inline float get_weight_b() noexcept { return GET_SETTING(RGB).weight.b; }
inline uint get_composite_p_red() { return GET_SETTING(RGB).frame_index.min; }
inline uint get_composite_p_blue() { return GET_SETTING(RGB).frame_index.max; }

inline void set_weight_r(double value) { SET_SETTING(RGB, weight.r, value); }
inline void set_weight_g(double value) { SET_SETTING(RGB, weight.g, value); }
inline void set_weight_b(double value) { SET_SETTING(RGB, weight.b, value); }
void set_weight_rgb(double r, double g, double b);
void set_rgb_p(int min, int max);

/*! \brief Modifies Frequency channel (p) Red (min) and Frequency channel (p) Blue (max) from ui values
 *
 * \param[in] composite_p_red the new value
 * \param[in] composite_p_blue the new value
 */
void set_composite_intervals(int composite_p_red, int composite_p_blue);
/*! \} */

#pragma endregion

#pragma region HSV Hue

/*! \name Hue Getters
 * \{
 */

/*! \brief Get the Hue minimum frequency
 *
 * \return the Hue minimum frequency
 */
inline uint get_composite_p_min_h() noexcept { return GET_SETTING(HSV).h.frame_index.min; }

/*! \brief Get the Hue maximum frequency
 *
 * \return the Hue maximum frequency
 */
inline uint get_composite_p_max_h() noexcept { return GET_SETTING(HSV).h.frame_index.max; }
inline float get_slider_h_threshold_min() noexcept { return GET_SETTING(HSV).h.slider_threshold.min; }
inline float get_slider_h_threshold_max() noexcept { return GET_SETTING(HSV).h.slider_threshold.max; }
inline float get_slider_h_shift_min() { return GET_SETTING(HSV).h.slider_shift.min; }
inline float get_slider_h_shift_max() { return GET_SETTING(HSV).h.slider_shift.max; }
/*! \} */

/*! \name Hue Setters
 * \{
 */

/*! \brief Set the Hue minimum frequency
 *
 * \param[in] value must be in range [0, composite_p_max_h]
 */
inline void set_composite_p_min_h(uint value)
{
    SET_SETTING(HSV, h.frame_index.min, value);
    pipe_refresh();
}

/*! \brief Set the Hue maximum frequency
 *
 * \param[in] value must be in range [composite_p_min_h, time_transformation_size]
 */
inline void set_composite_p_max_h(uint value)
{
    SET_SETTING(HSV, h.frame_index.max, value);
    pipe_refresh();
}

inline void set_slider_h_threshold_min(float value) { SET_SETTING(HSV, h.slider_threshold.min, value); }
inline void set_slider_h_threshold_max(float value) { SET_SETTING(HSV, h.slider_threshold.max, value); }
inline void set_slider_h_shift_min(float value) { SET_SETTING(HSV, h.slider_shift.min, value); }
inline void set_slider_h_shift_max(float value) { SET_SETTING(HSV, h.slider_shift.max, value); }
void set_composite_p_h(int min, int max);
/*! \} */

#pragma endregion

#pragma region HSV Saturation

/*! \name Saturation Getters
 * \{
 */

/*! \brief Get the Saturation minimum frequency
 *
 * \return the Saturation minimum frequency
 */
inline uint get_composite_p_min_s() noexcept { return GET_SETTING(HSV).s.frame_index.min; }

/*! \brief Get the Saturation maximum frequency
 *
 * \return the Saturation maximum frequency
 */
inline uint get_composite_p_max_s() noexcept { return GET_SETTING(HSV).s.frame_index.max; }

inline float get_slider_s_threshold_min() noexcept { return GET_SETTING(HSV).s.slider_threshold.min; }
inline float get_slider_s_threshold_max() noexcept { return GET_SETTING(HSV).s.slider_threshold.max; }

/*! \brief Return whether the Saturation frequency range is activated. If not activated, the min and max frequency of
 * the Hue channel will be used instead.
 *
 * Used whether to apply the following settings:
 * - \ref holovibes::api::get_composite_p_min_s "min frequency"
 * - \ref holovibes::api::get_composite_p_max_s "max frequency".
 *
 * \return true if the Saturation frequency range is activated
 */
inline bool get_composite_p_activated_s() noexcept { return GET_SETTING(HSV).s.frame_index.activated; }
/*! \} */

/*! \name Saturation Setters
 * \{
 */

/*! \brief Set the Saturation minimum frequency. Take place only if the \ref holovibes::api::get_composite_p_activated_s
 * "Saturation frequency range" is activated.
 *
 * \param[in] value must be in range [0, composite_p_max_s]
 */
inline void set_composite_p_min_s(uint value)
{
    SET_SETTING(HSV, s.frame_index.min, value);
    if (get_composite_p_activated_s())
        pipe_refresh();
}

/*! \brief Set the Saturation maximum frequency
 *
 * \param[in] value must be in range [composite_p_min_s, time_transformation_size]
 */
inline void set_composite_p_max_s(uint value)
{
    SET_SETTING(HSV, s.frame_index.max, value);
    if (get_composite_p_activated_s())
        pipe_refresh();
}

inline void set_slider_s_threshold_min(float value) { SET_SETTING(HSV, s.slider_threshold.min, value); }
inline void set_slider_s_threshold_max(float value) { SET_SETTING(HSV, s.slider_threshold.max, value); }

/*! \brief Set the Saturation frequency range activation. If not activated, the min and max frequency of
 * the Hue channel will be used instead.
 *
 * Used whether to apply the following settings:
 * - \ref holovibes::api::set_composite_p_min_s "min frequency"
 * - \ref holovibes::api::set_composite_p_max_s "max frequency".
 *
 * \param[in] value true to activate the Saturation frequency range.
 */
inline void set_composite_p_activated_s(bool value) { SET_SETTING(HSV, s.frame_index.activated, value); }
/*! \} */

#pragma endregion

#pragma region HSV Value

/*! \name Value Getters
 * \{
 */

/*! \brief Get the Value minimum frequency
 *
 * \return the Value minimum frequency
 */
inline uint get_composite_p_min_v() noexcept { return GET_SETTING(HSV).v.frame_index.min; }

/*! \brief Get the Value maximum frequency
 *
 * \return the Value maximum frequency
 */
inline uint get_composite_p_max_v() noexcept { return GET_SETTING(HSV).v.frame_index.max; }
inline float get_slider_v_threshold_min() noexcept { return GET_SETTING(HSV).v.slider_threshold.min; }
inline float get_slider_v_threshold_max() noexcept { return GET_SETTING(HSV).v.slider_threshold.max; }

/*! \brief Return whether the Value frequency range is activated. If not activated, the min and max frequency of
 * the Hue channel will be used instead.
 *
 * Used whether to apply the following settings:
 * - \ref holovibes::api::get_composite_p_min_v "min frequency"
 * - \ref holovibes::api::get_composite_p_max_v "max frequency".
 *
 * \return true if the Saturation frequency range is activated
 */
inline bool get_composite_p_activated_v() noexcept { return GET_SETTING(HSV).v.frame_index.activated; }
/*! \} */

/*! \name Value Setters
 * \{
 */

/*! \brief Set the Value minimum frequency
 *
 * \param[in] value must be in range [0, composite_p_max_v]
 */
inline void set_composite_p_min_v(uint value)
{
    SET_SETTING(HSV, v.frame_index.min, value);
    if (get_composite_p_activated_v())
        pipe_refresh();
}

/*! \brief Set the Value maximum frequency
 *
 * \param[in] value must be in range [composite_p_min_v, time_transformation_size]
 */
inline void set_composite_p_max_v(uint value)
{
    SET_SETTING(HSV, v.frame_index.max, value);
    if (get_composite_p_activated_v())
        pipe_refresh();
}

inline void set_slider_v_threshold_min(float value) { SET_SETTING(HSV, v.slider_threshold.min, value); }
inline void set_slider_v_threshold_max(float value) { SET_SETTING(HSV, v.slider_threshold.max, value); }

/*! \brief Set the Value frequency range activation. If not activated, the min and max frequency of
 * the Hue channel will be used instead.
 *
 * Used whether to apply the following settings:
 * - \ref holovibes::api::set_composite_p_min_v "min frequency"
 * - \ref holovibes::api::set_composite_p_max_v "max frequency".
 *
 * \param[in] value true to activate the Saturation frequency range.
 */
inline void set_composite_p_activated_v(bool value) { SET_SETTING(HSV, v.frame_index.activated, value); }
/*! \} */

#pragma endregion

} // namespace holovibes::api