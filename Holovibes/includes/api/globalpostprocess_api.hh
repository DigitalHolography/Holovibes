/*! \file
 *
 * \brief Regroup all functions used to interact with post processing operations done on the main image (the on on the
 * XY view panel). These operations are: convolution, registration and renormalization.
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{

#pragma region Registration Settings

/*! \brief Getter and Setter for the registration, triggered when the Registration button is clicked on the gui.
 * (Setter refreshes the pipe) */
inline bool get_registration_enabled() { return GET_SETTING(RegistrationEnabled); }

inline void set_registration_enabled(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    if (!api::get_fft_shift_enabled())
        set_fft_shift_enabled(value);

    UPDATE_SETTING(RegistrationEnabled, value);
    pipe_refresh();
}

inline float get_registration_zone() { return GET_SETTING(RegistrationZone); }
inline void set_registration_zone(float value) { UPDATE_SETTING(RegistrationZone, value); }

#pragma endregion

#pragma region Convolution Settings

inline std::vector<float> get_convo_matrix() { return GET_SETTING(ConvolutionMatrix); };
inline void set_convo_matrix(std::vector<float> value) { UPDATE_SETTING(ConvolutionMatrix, value); }

inline bool get_convolution_enabled() { return GET_SETTING(ConvolutionEnabled); }
inline void set_convolution_enabled(bool value) { UPDATE_SETTING(ConvolutionEnabled, value); }

inline bool get_divide_convolution_enabled() { return GET_SETTING(DivideConvolutionEnabled); }
inline void set_divide_convolution_enabled(bool value) { UPDATE_SETTING(DivideConvolutionEnabled, value); }

inline std::string get_convolution_file_name() { return GET_SETTING(ConvolutionFileName); }
inline void set_convolution_file_name(std::string value) { UPDATE_SETTING(ConvolutionFileName, value); }

#pragma endregion

#pragma region Renorm Settings

inline unsigned get_renorm_constant() { return GET_SETTING(RenormConstant); }
inline void set_renorm_constant(unsigned int value) { UPDATE_SETTING(RenormConstant, value); }

inline bool get_renorm_enabled() { return GET_SETTING(RenormEnabled); }
inline void set_renorm_enabled(bool value)
{
    UPDATE_SETTING(RenormEnabled, value);
    pipe_refresh();
}

#pragma endregion

#pragma region Registration

/*! \brief Set the new value of the registration zone for the circular mask. Range ]0, 1[.
 *  \param[in] value The new zone value.
 */
void update_registration_zone(float value);

#pragma endregion

#pragma region Convolution

/*! \brief Enables the divide convolution mode
 *
 * \param value the file containing the convolution's settings
 */
void enable_convolution(const std::string& file);

/*! \brief Loads convolution matrix from a given file
 *
 * \param file the file containing the convolution's settings
 */
void load_convolution_matrix(std::string filename);

/*! \brief Disables convolution
 *
 */
void disable_convolution();

/*! \brief Enable the divide convolution mode
 *
 * \param value true: enable, false: disable
 */
void set_divide_convolution(const bool value);

#pragma endregion

} // namespace holovibes::api