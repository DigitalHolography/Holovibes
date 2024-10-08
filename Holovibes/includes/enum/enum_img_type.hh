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
    Modulus = 0,    /*!< Modulus of the complex data */
    SquaredModulus, /*!<  Modulus taken to its square value */
    Argument,       /*!<  Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
    PhaseIncrease,  /*!<  Phase value, the conjugate between the phase of the last image and the previous one */
    Composite,      /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
    Moments_0,      /*!<  Displays the moments at order 0, 1 and 2. NOT IMPLEMENTED YET*/
    Moments_1,
    Moments_2,
};

// clang-format off
SERIALIZE_JSON_ENUM(ImgType, {
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
