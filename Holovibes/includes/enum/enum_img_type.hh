/*! \file
 *
 * \brief Enum for the different type of displaying images
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum ImgType
 *
 * \brief Displaying type of the image
 */
enum class ImgType
{
    Raw = 0,        /*!< The raw data */
    Modulus,        /*!< Modulus of the complex data */
    SquaredModulus, /*!<  Modulus taken to its square value */
    Argument,       /*!<  Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
    PhaseIncrease,  /*!<  Phase value, the conjugate between the phase of the last image and the previous one */
    Composite,      /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
    Moments_0,      /*!<  Displays the moment 0 (tensor_multiply of the output of the TT with fft_freq^0) */
    Moments_1,      /*!<  Displays the moment 1 (tensor_multiply of the output of the TT with fft_freq^1) */
    Moments_2,      /*!<  Displays the moment 2 (tensor_multiply of the output of the TT with fft_freq^2) */
};

// clang-format off
SERIALIZE_JSON_ENUM(ImgType, {
    {ImgType::Raw, "RAW"},
    {ImgType::Modulus, "MODULUS"},
    {ImgType::SquaredModulus, "SQUARED_MODULUS"},
    {ImgType::Argument, "ARGUMENT"},
    {ImgType::PhaseIncrease, "PHASE_INCREASE"},
    {ImgType::Composite, "COMPOSITE"},
    {ImgType::Moments_0, "MOMENTS_0"},
    {ImgType::Moments_1, "MOMENTS_1"},
    {ImgType::Moments_2, "MOMENTS_2"},
})
// clang-format on

} // namespace holovibes
