/*! \file filter2d_api.hh
 *
 * \brief Regroup all functions used to interact with the filter 2D settings.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

#pragma region Filter Settings

inline int get_filter2d_n1() { return GET_SETTING(Filter2dN1); }
void set_filter2d_n1(int value);

inline int get_filter2d_n2() { return GET_SETTING(Filter2dN2); }
void set_filter2d_n2(int value);

inline int get_filter2d_smooth_low() { return GET_SETTING(Filter2dSmoothLow); }
inline void set_filter2d_smooth_low(int value) { UPDATE_SETTING(Filter2dSmoothLow, value); }

inline int get_filter2d_smooth_high() { return GET_SETTING(Filter2dSmoothHigh); }
inline void set_filter2d_smooth_high(int value) { UPDATE_SETTING(Filter2dSmoothHigh, value); }

inline ViewWindow get_filter2d() { return GET_SETTING(Filter2d); }
inline void set_filter2d(ViewWindow value) noexcept { UPDATE_SETTING(Filter2d, value); }

inline bool get_filter2d_enabled() { return GET_SETTING(Filter2dEnabled); }
void set_filter2d_enabled(bool value);

#pragma endregion

#pragma region Filter File Settings

inline std::string get_filter_file_name() { return GET_SETTING(FilterFileName); }
inline void set_filter_file_name(std::string value) { UPDATE_SETTING(FilterFileName, value); }

inline bool get_filter_enabled() { return GET_SETTING(FilterEnabled); };
inline void set_filter_enabled(bool value) { UPDATE_SETTING(FilterEnabled, value); };

#pragma endregion

#pragma region Filter File

/*! \brief Gets the input filter
 *
 * \return the input filter
 */
std::vector<float> get_input_filter();

/*! \brief Sets the input filter
 *
 * \param value the new value of the input filter
 */
void set_input_filter(std::vector<float> value);

/*! \brief Loads the input filter
 *
 * \param file the file path
 */
void load_input_filter(const std::string& file);

/*! \brief Enables the input filter mode
 *
 * \param file the file containing the filter's settings or empty string to disable the filter
 */
void enable_filter(const std::string& file);

#pragma endregion

} // namespace holovibes::api