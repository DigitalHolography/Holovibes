/*! \file view_api.hh
 *
 * \brief Regroup all functions used to interact with the different view.
 *
 * Views are:
 * - Filter2D
 * - 3D Cuts
 * - Raw
 * - Lens
 * - Chart
 */
#pragma once

#include "API.hh"
#include "enum_window_kind.hh"

namespace holovibes::api
{

#pragma region View Settings

inline WindowKind get_current_window_type() { return GET_SETTING(CurrentWindow); }

/*! \brief Changes the focused windows
 *
 * \param index the index representing the window to select
 */
inline void change_window(const int index) { UPDATE_SETTING(CurrentWindow, static_cast<WindowKind>(index)); }

inline bool get_cuts_view_enabled() { return GET_SETTING(CutsViewEnabled); }
inline void set_cuts_view_enabled(bool value) { UPDATE_SETTING(CutsViewEnabled, value); }

inline bool get_filter2d_view_enabled() { return GET_SETTING(Filter2dViewEnabled); }
inline void set_filter2d_view_enabled(bool value) { UPDATE_SETTING(Filter2dViewEnabled, value); }

inline bool get_chart_display_enabled() { return GET_SETTING(ChartDisplayEnabled); }
inline void set_chart_display_enabled(bool value) { UPDATE_SETTING(ChartDisplayEnabled, value); }

inline bool get_lens_view_enabled() { return GET_SETTING(LensViewEnabled); }
inline void set_lens_view_enabled(bool value) { UPDATE_SETTING(LensViewEnabled, value); }

inline bool get_raw_view_enabled() { return GET_SETTING(RawViewEnabled); }
inline void set_raw_view_enabled(bool value) { UPDATE_SETTING(RawViewEnabled, value); }

inline float get_display_rate() { return GET_SETTING(DisplayRate); }
inline void set_display_rate(float value) { UPDATE_SETTING(DisplayRate, value); }

#pragma endregion

#pragma region Views Logic

/*! \brief Enables or Disables time transform cuts views
 *
 * \param[in] enabled true: enable, false: disable
 * \return true if correctly set
 */
bool set_3d_cuts_view(bool enabled);

/*! \brief Adds filter2d view
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_filter2d_view(bool enabled);

/*! \brief Start or stop the chart display
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_chart_display(bool enabled);

/*! \brief Adds or removes lens view.
 *
 * \param[in] enabled true: enable, false: disable
 */
void set_lens_view(bool enabled);

/*! \brief Adds or removes raw view

 * \param[in] enabled true: enable, false: disable
 */
void set_raw_view(bool enabled);

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