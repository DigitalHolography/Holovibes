/*! \file composite_api.hh
 *
 * \brief Regroup all functions used to interact with the composite settings.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

class CompositeApi : public IApi
{

  public:
#pragma region Composite

    /*! \brief Return the composite kind
     *
     * \return CompositeKind Either RGB or HSV
     */
    inline CompositeKind get_composite_kind() noexcept { return GET_SETTING(CompositeKind); }

    /*! \brief Set the composite kind
     *
     * \param[in] value The new composite kind
     */
    inline void set_composite_kind(CompositeKind value) { UPDATE_SETTING(CompositeKind, value); }

    /*! \brief Return the composite zone
     *
     * \return units::RectFd The composite zone
     */
    inline units::RectFd get_composite_zone() { return GET_SETTING(CompositeZone); }

    /*! \brief Set the composite zone
     *
     * \param[in] rect The new composite zone
     */
    inline void set_composite_zone(const units::RectFd& rect) { UPDATE_SETTING(CompositeZone, rect); }

#pragma endregion

#pragma region RGB Weights

    /*! \brief Return whether the RGB weights are automatically computed or not (similar to auto contrast).
     *
     * \return bool true if the composite auto weights is enabled
     */
    inline bool get_composite_auto_weights() noexcept { return GET_SETTING(CompositeAutoWeights); }

    /*! \brief Set whether the RGB weights are automatically computed or not (similar to auto contrast).
     *
     * \param[in] value enable or disable the composite auto weights
     */
    void set_composite_auto_weights(bool value);

    /*! \brief Returns the contribution of the Red channel to generate the final composite image.
     *
     * \return float the weight of the Red channel
     */
    inline float get_weight_r() noexcept { return GET_SETTING(RGB).weight.r; }

    /*! \brief Sets the contribution of the Green channel to generate the final composite image.
     *
     * \param[in] value the new weight of the Green channel
     */
    inline void set_weight_r(double value) { SET_SETTING(RGB, weight.r, value); }

    /*! \brief Returns the contribution of the Green channel to generate the final composite image.
     *
     * \return float the weight of the Green channel
     */
    inline float get_weight_g() noexcept { return GET_SETTING(RGB).weight.g; }

    /*! \brief Sets the contribution of the Green channel to generate the final composite image.
     *
     * \param[in] value the new weight of the Green channel
     */
    inline void set_weight_g(double value) { SET_SETTING(RGB, weight.g, value); }

    /*! \brief Returns the contribution of the Blue channel to generate the final composite image.
     *
     * \return float the weight of the Blue channel
     */
    inline float get_weight_b() noexcept { return GET_SETTING(RGB).weight.b; }

    /*! \brief Sets the contribution of the Blue channel to generate the final composite image.
     *
     * \param[in] value the new weight of the Blue channel
     */
    inline void set_weight_b(double value) { SET_SETTING(RGB, weight.b, value); }

    /*! \brief Sets the contribution of the RGB channels to generate the final composite image.
     *
     * \param[in] r the new weight of the Red channel
     * \param[in] g the new weight of the Green channel
     * \param[in] b the new weight of the Blue channel
     */
    void set_weight_rgb(double r, double g, double b);

#pragma endregion

#pragma region RGB

    /*! \brief Returns the min accumulation frequency. Is in range [0, `time_transformation_size -
     * get_composite_p_blue - 1`].
     *
     * When generating the composite image, the frequency ranging (resulting from the time transformation) between
     * [`get_composite_p_red`, `get_composite_p_red + get_composite_p_blue`] will be accumulated into one image.
     *
     * \return uint the min accumulation frequency
     */
    inline uint get_composite_p_red() { return GET_SETTING(RGB).frame_index.min; }

    /*! \brief Returns the max accumulation frequency. Is in range 0, `time_transformation_size -
     * get_composite_p_red - 1`].
     *
     * When generating the composite image, the frequency ranging (resulting from the time transformation) between
     * [`get_composite_p_red`, `get_composite_p_red + get_composite_p_blue`] will be accumulated into one image.
     *
     * \return uint the max accumulation frequency
     */
    inline uint get_composite_p_blue() { return GET_SETTING(RGB).frame_index.max; }

    /*! \brief Set the min and max accumulation frequency.
     *
     * When generating the composite image, the frequency ranging (resulting from the time transformation) between
     * [`get_composite_p_red`, `get_composite_p_red + get_composite_p_blue`] will be accumulated into one image.
     *
     * \param[in] red the new min accumulation frequency
     * \param[in] blue the new max accumulation frequency
     */
    void set_rgb_p(int red, int blue);

#pragma endregion

#pragma region HSV

    /*! \brief Return whether the Z FFT shift is enabled or not.
     *
     * \return bool true if the Z FFT shift is enabled
     */

    inline bool get_z_fft_shift() noexcept { return GET_SETTING(ZFFTShift); }

    /*! \brief Set whether the Z FFT shift is enabled or not.
     *
     * \param[in] checked true to enable the Z FFT shift
     */
    inline void set_z_fft_shift(bool checked) { UPDATE_SETTING(ZFFTShift, checked); }

#pragma endregion

#pragma region HSV Hue Freq.

    /*! \brief Get the Hue minimum frequency
     *
     * \return uint the Hue minimum frequency
     */
    inline uint get_composite_p_min_h() noexcept { return GET_SETTING(HSV).h.frame_index.min; }

    /*! \brief Set the Hue minimum frequency
     *
     * \param[in] value must be in range [0, composite_p_max_h]
     */
    void set_composite_p_min_h(uint value);

    /*! \brief Get the Hue maximum frequency
     *
     * \return uint the Hue maximum frequency
     */
    inline uint get_composite_p_max_h() noexcept { return GET_SETTING(HSV).h.frame_index.max; }

    /*! \brief Set the Hue maximum frequency
     *
     * \param[in] value must be in range [composite_p_min_h, time_transformation_size]
     */
    void set_composite_p_max_h(uint value);

    /*! \brief Set the Hue frequency range accumulation.
     *
     * \param[in] min must be in range [0, composite_p_max_h]
     * \param[in] max must be in range [composite_p_min_h, time_transformation_size]
     */
    void set_composite_p_h(int min, int max);

#pragma endregion

#pragma region HSV Hue

    /*! \brief Gets the Hue threshold minimum value
     *
     * \return float the Hue threshold minimum value
     */
    inline float get_slider_h_threshold_min() noexcept { return GET_SETTING(HSV).h.slider_threshold.min; }

    /*! \brief Sets the Hue threshold minimum value
     *
     * \param[in] value the new Hue threshold minimum value
     */
    inline void set_slider_h_threshold_min(float value) { SET_SETTING(HSV, h.slider_threshold.min, value); }

    /*! \brief Gets the Hue threshold maximum value
     *
     * \return float the Hue threshold maximum value
     */
    inline float get_slider_h_threshold_max() noexcept { return GET_SETTING(HSV).h.slider_threshold.max; }

    /*! \brief Sets the Hue threshold maximum value
     *
     * \param[in] value the new Hue threshold maximum value
     */
    inline void set_slider_h_threshold_max(float value) { SET_SETTING(HSV, h.slider_threshold.max, value); }

    /*! \brief Gets the Hue shift minimum value
     *
     * \return float the Hue shift minimum value
     */
    inline float get_slider_h_shift_min() { return GET_SETTING(HSV).h.slider_shift.min; }

    /*! \brief Sets the Hue shift minimum value
     *
     * \param[in] value the new Hue shift minimum value
     */
    inline void set_slider_h_shift_min(float value) { SET_SETTING(HSV, h.slider_shift.min, value); }

    /*! \brief Gets the Hue shift maximum value
     *
     * \return float the Hue shift maximum value
     */
    inline float get_slider_h_shift_max() { return GET_SETTING(HSV).h.slider_shift.max; }

    /*! \brief Sets the Hue shift maximum value
     *
     * \param[in] value the new Hue shift maximum value
     */
    inline void set_slider_h_shift_max(float value) { SET_SETTING(HSV, h.slider_shift.max, value); }

#pragma endregion

#pragma region HSV Saturation Freq.

    /*! \brief Get the Saturation minimum frequency
     *
     * \return the Saturation minimum frequency
     */
    inline uint get_composite_p_min_s() noexcept { return GET_SETTING(HSV).s.frame_index.min; }

    /*! \brief Set the Saturation minimum frequency. Take place only if the \ref
     * holovibes::api::get_composite_p_activated_s "Saturation frequency range" is activated.
     *
     * \param[in] value must be in range [0, composite_p_max_s]
     */
    void set_composite_p_min_s(uint value);

    /*! \brief Get the Saturation maximum frequency
     *
     * \return uint the Saturation maximum frequency
     */
    inline uint get_composite_p_max_s() noexcept { return GET_SETTING(HSV).s.frame_index.max; }

    /*! \brief Set the Saturation maximum frequency
     *
     * \param[in] value must be in range [composite_p_min_s, time_transformation_size]
     */
    void set_composite_p_max_s(uint value);

    /*! \brief Return whether the Saturation frequency range is activated. If not activated, the min and max frequency
     * of the Hue channel will be used instead.
     *
     * Used whether to apply the following settings:
     * - \ref holovibes::api::get_composite_p_min_s "min frequency"
     * - \ref holovibes::api::get_composite_p_max_s "max frequency".
     *
     * \return bool true if the Saturation frequency range is activated
     */
    inline bool get_composite_p_activated_s() noexcept { return GET_SETTING(HSV).s.frame_index.activated; }

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

#pragma endregion

#pragma region HSV Saturation

    /*! \brief Get the Saturation threshold minimum value
     *
     * \return float the Saturation threshold minimum value
     */
    inline float get_slider_s_threshold_min() noexcept { return GET_SETTING(HSV).s.slider_threshold.min; }

    /*! \brief Set the Saturation threshold minimum value
     *
     * \param[in] value the new Saturation threshold minimum value
     */
    inline void set_slider_s_threshold_min(float value) { SET_SETTING(HSV, s.slider_threshold.min, value); }

    /*! \brief Get the Saturation threshold maximum value
     *
     * \return float the Saturation threshold maximum value
     */
    inline float get_slider_s_threshold_max() noexcept { return GET_SETTING(HSV).s.slider_threshold.max; }

    /*! \brief Set the Saturation threshold maximum value
     *
     * \param[in] value the new Saturation threshold maximum value
     */
    inline void set_slider_s_threshold_max(float value) { SET_SETTING(HSV, s.slider_threshold.max, value); }

#pragma endregion

#pragma region HSV Value Freq.

    /*! \brief Get the Value minimum frequency
     *
     * \return uint the Value minimum frequency
     */
    inline uint get_composite_p_min_v() noexcept { return GET_SETTING(HSV).v.frame_index.min; }

    /*! \brief Set the Value minimum frequency
     *
     * \param[in] value must be in range [0, composite_p_max_v]
     */
    void set_composite_p_min_v(uint value);

    /*! \brief Get the Value maximum frequency
     *
     * \return uint the Value maximum frequency
     */
    inline uint get_composite_p_max_v() noexcept { return GET_SETTING(HSV).v.frame_index.max; }

    /*! \brief Set the Value maximum frequency
     *
     * \param[in] value must be in range [composite_p_min_v, time_transformation_size]
     */
    void set_composite_p_max_v(uint value);

    /*! \brief Return whether the Value frequency range is activated. If not activated, the min and max frequency of
     * the Hue channel will be used instead.
     *
     * Used whether to apply the following settings:
     * - \ref holovibes::api::get_composite_p_min_v "min frequency"
     * - \ref holovibes::api::get_composite_p_max_v "max frequency".
     *
     * \return bool true if the Saturation frequency range is activated
     */
    inline bool get_composite_p_activated_v() noexcept { return GET_SETTING(HSV).v.frame_index.activated; }

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

#pragma endregion

#pragma region HSV Value

    /*! \brief Get the Value threshold minimum value
     *
     * \return float the Value threshold minimum value
     */
    inline float get_slider_v_threshold_min() noexcept { return GET_SETTING(HSV).v.slider_threshold.min; }

    /*! \brief Set the Value threshold minimum value
     *
     * \param[in] value the new Value threshold minimum value
     */
    inline void set_slider_v_threshold_min(float value) { SET_SETTING(HSV, v.slider_threshold.min, value); }

    /*! \brief Get the Value threshold maximum value
     *
     * \return float the Value threshold maximum value
     */
    inline float get_slider_v_threshold_max() noexcept { return GET_SETTING(HSV).v.slider_threshold.max; }

    /*! \brief Set the Value threshold maximum value
     *
     * \param[in] value the new Value threshold maximum value
     */
    inline void set_slider_v_threshold_max(float value) { SET_SETTING(HSV, v.slider_threshold.max, value); }

#pragma endregion
};

} // namespace holovibes::api