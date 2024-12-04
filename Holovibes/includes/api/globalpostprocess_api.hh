/*! \file globalpostprocess_api.hh
 *
 * \brief Regroup all functions used to interact with post processing operations done on the main image (the on on the
 * XY view panel). These operations are: convolution, registration and renormalization.
 */
#pragma once

#include "API.hh"

namespace holovibes::api
{

#pragma region Registration

/*! \brief Returns whether the registration is enabled or not. The registration is a post-processing step used to
 * correct motion artifacts.
 *
 * \return bool true if registration is enabled, false otherwise
 */
inline bool get_registration_enabled() { return GET_SETTING(RegistrationEnabled); }

/*! \brief Enables or disables the registration. The registration is a post-processing step used to correct motion
 * artifacts.
 *
 * \param[in] value true: enable, false: disable
 */
void set_registration_enabled(bool value);

/*! \brief Returns the radius of the circular mask used for the registration. Is in range ]0, 1[.
 *
 * \return float The registration zone value
 */
inline float get_registration_zone() { return GET_SETTING(RegistrationZone); }

/*! \brief Sets the radius of the circular mask used for the registration. Must be in range ]0, 1[.
 *
 *  \param[in] value The new zone value.
 */
inline void set_registration_zone(float value) { UPDATE_SETTING(RegistrationZone, value); }

/*! \brief Set the new value of the registration zone for the circular mask. Must be in range ]0, 1[.
 *
 *  \param[in] value The new zone value.
 */
void update_registration_zone(float value);

#pragma endregion

#pragma region Renormalizaiton

/*! \brief Returns whether the renormalization is enabled or not. The renormalization is a post-processing step used to
 * correct the intensity of the image.
 *
 * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of the
 * image to renormalize.
 *
 * \return bool true if renormalization is enabled, false otherwise
 */
inline bool get_renorm_enabled() { return GET_SETTING(RenormEnabled); }

/*! \brief Enables or disables the renormalization. The renormalization is a post-processing step used to correct the
 * intensity of the image.
 *
 * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of the
 * image to renormalize.
 *
 * \param[in] value true: enable, false: disable
 */
void set_renorm_enabled(bool value);

/*! \brief Returns the renormalization constant. The renormalization is a post-processing step used to correct the
 * intensity of the image.
 *
 * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of the
 * image to renormalize.
 *
 * \return unsigned The renormalization constant
 */
inline unsigned get_renorm_constant() { return GET_SETTING(RenormConstant); }

/*! \brief Sets the renormalization constant. The renormalization is a post-processing step used to correct the
 * intensity of the image.
 *
 * The formula used for the renormalization is: `px =  px * 2^(renorm_constant) / mean`. Where mean is the mean of the
 * image to renormalize.
 *
 * \param[in] value The new renormalization constant
 */
inline void set_renorm_constant(unsigned int value) { UPDATE_SETTING(RenormConstant, value); }

#pragma endregion

#pragma region Conv Matrix

/*! \brief Returns the convolution matrix/kernel used for the convolution post-processing step.
 *
 * \return std::vector<float> The convolution matrix/kernel
 */
inline std::vector<float> get_convo_matrix() { return GET_SETTING(ConvolutionMatrix); };

/*! \brief Sets the convolution matrix/kernel used for the convolution post-processing step.
 *
 * \param[in] value The new convolution matrix/kernel
 */
inline void set_convo_matrix(std::vector<float> value) { UPDATE_SETTING(ConvolutionMatrix, value); }

/*! \brief Loads a convolution matrix from a given file
 *
 * \param[in] file the file containing the convolution's settings
 */
void load_convolution_matrix(std::string filename);

#pragma endregion

#pragma region Conv Divide

/*! \brief Returns whether the original image should be divided by the convolutioned one or not.
 *
 * The calculation is: `out = in / conv(in)`
 *
 * \return bool true if divide convolution mode is enabled, false otherwise
 */
inline bool get_divide_convolution_enabled() { return GET_SETTING(DivideConvolutionEnabled); }

/*! \brief Sets whether the original image should be divided by the convolutioned one or not.
 *
 * The calculation is: `out = in / conv(in)`
 *
 * \param[in] value true: enable, false: disable
 */
void set_divide_convolution_enabled(const bool value);

#pragma endregion

#pragma region Convolution

/*! \brief Returns whether the convolution is enabled or not.
 *
 * \return bool true if enabled, false otherwise
 */
inline bool get_convolution_enabled() { return GET_SETTING(ConvolutionEnabled); }

/*! \brief Enables the convolution and loads the convolution matrix/kernel from the given file
 *
 * \param[in] file The file containing the convolution matrix/kernel
 */
void enable_convolution(const std::string& file);

/*! \brief Disables the convolution */
void disable_convolution();

/*! \brief Returns the path of the file containing the convolution matrix/kernel
 *
 * \return std::string The path of the file
 */
inline std::string get_convolution_file_name() { return GET_SETTING(ConvolutionFileName); }

/*! \brief Sets the path of the file containing the convolution matrix/kernel
 *
 * \param[in] value The path of the file
 */
inline void set_convolution_file_name(std::string value) { UPDATE_SETTING(ConvolutionFileName, value); }

#pragma endregion

} // namespace holovibes::api