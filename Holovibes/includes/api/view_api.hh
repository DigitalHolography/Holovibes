/*! \file view_api.hh
 *
 * \brief Regroup all functions used to interact with the different view and view related settings.
 *
 * Views are:
 * - 3D Cuts
 * - Filter2D
 * - Chart
 * - Lens
 * - Raw
 */
#pragma once

#include "API.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

#pragma region Focused Window

/*! \brief Returns the type of the focused window. This setting is only useful if you use functions overload that does
 * not take a WindowKind as parameter (for contrast, log and other window specific computation).
 *
 * \return WindowKind the current window type
 */
inline WindowKind get_current_window_type() { return GET_SETTING(CurrentWindow); }

/*! \brief Changes the focused window. This function is only useful if you use functions overload that does
 * not take a WindowKind as parameter (for contrast, log and other window specific computation).
 *
 * \param[in] kind the new window type
 */
inline void change_window(const WindowKind kind) { UPDATE_SETTING(CurrentWindow, kind); }

#pragma endregion

#pragma region 3D Cuts View

/*! \brief Returns whether the 3D cuts view are enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_cuts_view_enabled() { return GET_SETTING(CutsViewEnabled); }

/*! \brief Enables or Disables time transform cuts views
 *
 * \param[in] enabled true: enable, false: disable
 * \return true if correctly set
 */
bool set_3d_cuts_view(bool enabled);

#pragma endregion

#pragma region Filter2D View

/*! \brief Returns whether the 2D filter view is enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_filter2d_view_enabled() { return GET_SETTING(Filter2dViewEnabled); }

/*! \brief Adds filter2d view
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_filter2d_view(bool enabled);

#pragma endregion

#pragma region Chart View

/*! \brief Returns whether the chart display is enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_chart_display_enabled() { return GET_SETTING(ChartDisplayEnabled); }

/*! \brief Start or stop the chart display
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_chart_display(bool enabled);

#pragma endregion

#pragma region Lens View

/*! \brief Returns whether the lens view is enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_lens_view_enabled() { return GET_SETTING(LensViewEnabled); }

/*! \brief Adds or removes lens view.
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_lens_view(bool enabled);

#pragma endregion

#pragma region Raw View

/*! \brief Returns whether the raw view is enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_raw_view_enabled() { return GET_SETTING(RawViewEnabled); }

/*! \brief Adds or removes raw view

 * \param[in] enabled true: enable, false: disable
 */
void set_raw_view(bool enabled);

#pragma endregion

#pragma region Display Rate

/*! \brief Returns the display rate of the view.
 *
 * \return float the display rate
 */
inline float get_display_rate() { return GET_SETTING(DisplayRate); }

/*! \brief Sets the display rate of the view.
 *
 * \param[in] value the new display rate
 */
inline void set_display_rate(float value) { UPDATE_SETTING(DisplayRate, value); }

#pragma endregion

#pragma region Last Image

void* get_raw_last_image();      // get_input_queue().get()
void* get_raw_view_last_image(); // get_input_queue().get()
void* get_hologram_last_image(); // get_gpu_output_queue().get()
void* get_lens_last_image();     // api::get_compute_pipe()->get_lens_queue().get()
void* get_xz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(0).get()
void* get_yz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(1).get()
void* get_filter2d_last_image(); // api::get_compute_pipe()->get_filter2d_view_queue().get()
void* get_chart_last_image();    // api::get_compute_pipe()->get_chart_display_queue().get()

#pragma endregion

} // namespace holovibes::api