
/*! \file
 *
 * \brief Enum for the different type of displaying images
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum ImageTypeEnum
 *
 * \brief Displaying type of the image
 */
enum class ImageTypeEnum
{
    Modulus = 0,    /*!< Modulus of the complex data */
    SquaredModulus, /*!<  Modulus taken to its square value */
    Argument,       /*!<  Phase (angle) value of the complex pixel c, computed with atan(Im(c)/Re(c)) */
    PhaseIncrease,  /*!<  Phase value, the conjugate between the phase of the last image and the previous one */
    Composite       /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
};

// clang-format off

SERIALIZE_JSON_ENUM(ImageTypeEnum, {
    {ImageTypeEnum::Modulus, "MODULUS"},
    {ImageTypeEnum::SquaredModulus, "SQUARED_MODULUS"},
    {ImageTypeEnum::Argument, "ARGUMENT"},
    {ImageTypeEnum::PhaseIncrease, "PHASE_INCREASE"},
    {ImageTypeEnum::Composite, "COMPOSITE"},
})

// clang-format on

inline std::ostream& operator<<(std::ostream& os, const ImageTypeEnum& value) { return os << json{value}; }

} // namespace holovibes
