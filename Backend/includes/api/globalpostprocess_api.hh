/*! \file globalpostprocess_api.hh
 *
 * \brief Regroup all functions used to interact with post processing operations done on the main image (the one inside
 * the gpu_output_buffer). These operations are: convolution, registration and renormalization.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

/*! \class GlobalPostProcessApi
 *
 * \brief Regroup all functions to interact with post processing operations done on the main image (the one inside the
 * gpu_output_buffer). These operations are: convolution (file path, matrix, divide), registration and renormalization.
 */
class GlobalPostProcessApi : public IApi
{

  public:
    GlobalPostProcessApi(const Api* api)
        : IApi(api)
    {
    }

#pragma region Registration

    /*! \brief Returns whether the registration is enabled or not. The registration is a post-processing step used to
     * correct motion artifacts.
     *
     * \return bool true if registration is enabled, false otherwise
     */
    inline bool get_registration_enabled() const { return GET_SETTING(RegistrationEnabled); }

    /*! \brief Enables or disables the registration. The registration is a post-processing step used to correct motion
     * artifacts.
     *
     * \param[in] value true: enable, false: disable
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, OK otherwise
     */
    ApiCode set_registration_enabled(bool value) const;

    /*! \brief Returns the radius of the circular mask used for the registration. Is in range ]0, 1[.
     *
     * \return float The registration zone value
     */
    inline float get_registration_zone() const { return GET_SETTING(RegistrationZone); }

    /*! \brief Sets the radius of the circular mask used for the registration. Must be in range ]0, 1[.
     *
     *  \param[in] value The new zone value.
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if registration is not enabled or if not in range ]0, 1[, OK otherwise
     */
    ApiCode set_registration_zone(float value) const;

#pragma endregion

#pragma region Renormalizaiton

    /*! \brief Returns whether the renormalization is enabled or not. The renormalization is a post-processing step used
     * to correct the intensity of the image.
     *
     * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of
     * the image to renormalize.
     *
     * \return bool true if renormalization is enabled, false otherwise
     */
    inline bool get_renorm_enabled() const { return GET_SETTING(RenormEnabled); }

    /*! \brief Enables or disables the renormalization. The renormalization is a post-processing step used to correct
     * the intensity of the image.
     *
     * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of
     * the image to renormalize.
     *
     * \param[in] value true: enable, false: disable
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, OK otherwise
     */
    ApiCode set_renorm_enabled(bool value) const;

    /*! \brief Returns the renormalization constant. The renormalization is a post-processing step used to correct the
     * intensity of the image.
     *
     * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of
     * the image to renormalize.
     *
     * \return unsigned The renormalization constant
     */
    inline unsigned get_renorm_constant() const { return GET_SETTING(RenormConstant); }

    /*! \brief Sets the renormalization constant. The renormalization is a post-processing step used to correct the
     * intensity of the image.
     *
     * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of
     * the image to renormalize.
     *
     * \param[in] value The new renormalization constant
     *
     * \return ApiCode NO_CHANGE if the value is the same, OK otherwise
     */
    ApiCode set_renorm_constant(unsigned int value) const;

#pragma endregion

#pragma region Conv Matrix

    /*! \brief Returns the convolution matrix/kernel used for the convolution post-processing step.
     *
     * \return std::vector<float> The convolution matrix/kernel
     */
    inline std::vector<float> get_convo_matrix() const { return GET_SETTING(ConvolutionMatrix); };

#pragma endregion

#pragma region Conv Divide

    /*! \brief Returns whether the original image should be divided by the convolutioned one or not.
     *
     * The calculation is: `out = in / conv(in)`
     *
     * \return bool true if divide convolution mode is enabled, false otherwise
     */
    inline bool get_divide_convolution_enabled() const { return GET_SETTING(DivideConvolutionEnabled); }

    /*! \brief Sets whether the original image should be divided by the convolutioned one or not.
     *
     * The calculation is: `out = in / conv(in)`
     *
     * \param[in] value true: enable, false: disable
     *
     * \return ApiCode NO_CHANGE if the value is the same, WRONG_COMP_MODE if the computation mode is Raw, INVALID_VALUE
     * if no convolution is loaded, OK otherwise
     */
    ApiCode set_divide_convolution_enabled(const bool value) const;

#pragma endregion

#pragma region Convolution

    /*! \brief Enables the convolution and loads the convolution matrix/kernel from the given file
     *
     * \param[in] file The file containing the convolution matrix/kernel
     *
     * \return ApiCode OK if the convolution was enabled, WRONG_COMP_MODE if we are in Raw mode or FAILURE if the matrix
     * could not be loaded.
     */
    ApiCode enable_convolution(const std::string& file) const;

    /*! \brief Returns the path of the file containing the convolution matrix/kernel
     *
     * \return std::string The path of the file
     */
    inline std::string get_convolution_file_name() const { return GET_SETTING(ConvolutionFileName); }

#pragma endregion

  private:
    /*!
     * \brief Loads a convolution matrix from a file
     *
     * This function is a tool / util supposed to be called by other functions
     *
     * \param[in] file The name of the file to load the matrix from. NOT A FULL PATH
     *
     * \return std::vector<float> The convolution matrix or an empty vector in case of error.
     */
    std::vector<float> load_convolution_matrix(const std::string& file) const;
};

} // namespace holovibes::api