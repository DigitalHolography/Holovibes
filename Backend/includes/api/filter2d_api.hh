/*! \file filter2d_api.hh
 *
 * \brief Regroup all functions used to interact with the filter 2D settings.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

/*! \class Filter2dApi
 *
 * \brief Regroup all functions related to filter 2D settings.
 */
class Filter2dApi : public IApi
{

  public:
    Filter2dApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Filter 2D

    /*! \brief Returns whether the filter 2D is enabled
     *
     * \return bool true if the filter 2D is enabled, false otherwise
     */
    inline bool get_filter2d_enabled() const { return GET_SETTING(Filter2dEnabled); }

    /*! \brief Enables or disables the filter 2D
     *
     * \param[in] value true to enable the filter 2D, false to disable it
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if mode is Raw, OK otherwise
     */
    ApiCode set_filter2d_enabled(bool value) const;

#pragma endregion

#pragma region Filter First Circle

    /*! \brief Returns the radius of the first circle used to construct the mask for the filter 2D. THe filter2D is a
     * bandpass filter done before the Spatial Transformation.
     *
     * \return int the radius of the first circle.
     */
    inline int get_filter2d_n1() const { return GET_SETTING(Filter2dN1); }

    /*! \brief Sets the radius of the first circle used to construct the mask for the filter 2D. THe filter2D is a
     * bandpass filter done before the Spatial Transformation.
     *
     * \param[in] value the new value of the radius. In range [0, Filter2dN2[
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if mode is Raw, INVALID_VALUE if the value is
     * not in range [0, Filter2dN2[, OK otherwise
     */
    ApiCode set_filter2d_n1(int value) const;

    /*! \brief Returns the smooth size of the second circle used to construct the mask for the filter 2D.
     *
     * \return int the smooth size of the second circle
     */
    inline int get_filter2d_smooth_high() const { return GET_SETTING(Filter2dSmoothHigh); }

    /*! \brief Sets the smooth size of the second circle used to construct the mask for the filter 2D.
     *
     * \param[in] value the new value of the smooth size.
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if mode is Raw, OK otherwise
     */
    ApiCode set_filter2d_smooth_high(int value) const;

#pragma endregion

#pragma region Filter Second Circle

    /*! \brief Returns the radius of the second circle used to construct the mask for the filter 2D. THe filter2D is a
     * bandpass filter done before the Spatial Transformation.
     *
     * \return int the radius of the second circle
     */
    inline int get_filter2d_n2() const { return GET_SETTING(Filter2dN2); }

    /*! \brief Sets the radius of the second circle used to construct the mask for the filter 2D. THe filter2D is a
     * bandpass filter done before the Spatial Transformation.
     *
     * \param[in] value the new value of the radius. In range ]Filter2dN1, +inf[
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if mode is Raw, INVALID_VALUE if the value is
     * not in range ]Filter2dN1, +inf[, OK otherwise
     */
    ApiCode set_filter2d_n2(int value) const;

    /*! \brief Returns the smooth size of the first circle used to construct the mask for the filter 2D.
     *
     * \return int the smooth size of the first circle
     */
    inline int get_filter2d_smooth_low() const { return GET_SETTING(Filter2dSmoothLow); }

    /*! \brief Sets the smooth size of the first circle used to construct the mask for the filter 2D.
     *
     * \param[in] value the new value of the smooth size
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if mode is Raw, OK otherwise
     */
    ApiCode set_filter2d_smooth_low(int value) const;

#pragma endregion

#pragma region Filter File Name

    /*! \brief Returns the file that will be used as a filter
     *
     * \return std::string the path of the file
     */
    inline std::string get_filter_file_name() const { return GET_SETTING(FilterFileName); }

#pragma endregion

#pragma region Filter File

    /*! \brief Gets the input filter
     *
     * \return std::vector<float> the input filter
     */
    inline std::vector<float> get_input_filter() const { return GET_SETTING(InputFilter); }

    /*! \brief Enables the input filter mode
     *
     * \param[in] file the file containing the filter's settings or empty string to disable the filter
     *
     * \return ApiCode the status of the operation: OK if successful, NOT_STARTED if the computation is not started
     */
    ApiCode enable_filter(const std::string& file) const;

  private:
    /*! \brief Loads the input filter
     *
     * \param[in] file the file path
     * \return std::vector<float> the input filter or an empty vector in case of error
     */
    std::vector<float> load_input_filter(const std::string& file) const;

#pragma endregion
};

} // namespace holovibes::api