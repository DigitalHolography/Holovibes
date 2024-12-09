/*! \file filter2d_api.hh
 *
 * \brief Regroup all functions used to interact with the filter 2D settings.
 */
#pragma once

#include "API.hh"

namespace holovibes::api
{

#pragma region Filter 2D

/*! \brief Returns whether the filter 2D is enabled
 *
 * \return bool true if the filter 2D is enabled, false otherwise
 */
inline bool get_filter2d_enabled() { return GET_SETTING(Filter2dEnabled); }

/*! \brief Enables or disables the filter 2D
 *
 * \param[in] value true to enable the filter 2D, false to disable it
 */
void set_filter2d_enabled(bool value);

#pragma endregion

#pragma region Filter First Circle

/*! \brief Returns the radius of the first circle used to construct the mask for the filter 2D. THe filter2D is a
 * bandpass filter done before the Spatial Transformation.
 *
 * \return int the radius of the first circle
 */
inline int get_filter2d_n1() { return GET_SETTING(Filter2dN1); }

/*! \brief Sets the radius of the first circle used to construct the mask for the filter 2D. THe filter2D is a bandpass
 * filter done before the Spatial Transformation.
 *
 * \param[in] value the new value of the radius
 */
void set_filter2d_n1(int value);

/*! \brief Returns the smooth size of the second circle used to construct the mask for the filter 2D.
 *
 * \return int the smooth size of the second circle
 */
inline int get_filter2d_smooth_high() { return GET_SETTING(Filter2dSmoothHigh); }

/*! \brief Sets the smooth size of the second circle used to construct the mask for the filter 2D.
 *
 * \param[in] value the new value of the smooth size
 */
inline void set_filter2d_smooth_high(int value) { UPDATE_SETTING(Filter2dSmoothHigh, value); }

#pragma endregion

#pragma region Filter Second Circle

/*! \brief Returns the radius of the second circle used to construct the mask for the filter 2D. THe filter2D is a
 * bandpass filter done before the Spatial Transformation.
 *
 * \return int the radius of the second circle
 */
inline int get_filter2d_n2() { return GET_SETTING(Filter2dN2); }

/*! \brief Sets the radius of the second circle used to construct the mask for the filter 2D. THe filter2D is a bandpass
 * filter done before the Spatial Transformation.
 *
 * \param[in] value the new value of the radius
 */
void set_filter2d_n2(int value);

/*! \brief Returns the smooth size of the first circle used to construct the mask for the filter 2D.
 *
 * \return int the smooth size of the first circle
 */
inline int get_filter2d_smooth_low() { return GET_SETTING(Filter2dSmoothLow); }

/*! \brief Sets the smooth size of the first circle used to construct the mask for the filter 2D.
 *
 * \param[in] value the new value of the smooth size
 */
inline void set_filter2d_smooth_low(int value) { UPDATE_SETTING(Filter2dSmoothLow, value); }

#pragma endregion

#pragma region Filter File Name

/*! \brief Returns the file that will be used as a filter
 *
 * \return std::string the path of the file
 */
inline std::string get_filter_file_name() { return GET_SETTING(FilterFileName); }

/*! \brief Sets the file that will be used as a filter
 *
 * \param[in] value the path of the file
 */
inline void set_filter_file_name(std::string value) { UPDATE_SETTING(FilterFileName, value); }

#pragma endregion

#pragma region Filter File

/*! \brief Returns whether a file filter is loaded
 *
 * \return bool true if a file filter is loaded, false otherwise
 */
inline bool get_filter_enabled() { return GET_SETTING(FilterEnabled); };

/*! \brief Gets the input filter
 *
 * \return std::vector<float> the input filter
 */
inline std::vector<float> get_input_filter() { return GET_SETTING(InputFilter); }

/*! \brief Sets the input filter
 *
 * \param[in] value the new value of the input filter
 */
inline void set_input_filter(std::vector<float> value) { UPDATE_SETTING(InputFilter, value); }

/*! \brief Loads the input filter
 *
 * \param[in] file the file path
 */
void load_input_filter(const std::string& file);

/*! \brief Enables the input filter mode
 *
 * \param[in] file the file containing the filter's settings or empty string to disable the filter
 */
void enable_filter(const std::string& file);

#pragma endregion

} // namespace holovibes::api