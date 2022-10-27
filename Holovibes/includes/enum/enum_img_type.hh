
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
SERIALIZE_JSON_ENUM(ImgTypeEnum, {
    {ImgTypeEnum::Modulus, "MODULUS"},
    {ImgTypeEnum::SquaredModulus, "SQUAREDMODULUS"},
    {ImgTypeEnum::Argument, "ARGUMENT"},
    {ImgTypeEnum::PhaseIncrease, "PHASEINCREASE"},
    {ImgTypeEnum::Composite, "COMPOSITE"},
})
inline std::string img_type_to_string(ImageTypeEnum value) { return _internal::img_type_to_string.at(value); }

inline ImageTypeEnum img_type_from_string(const std::string& in) { return _internal::string_to_img_type.at(in); }

inline std::ostream& operator<<(std::ostream& os, holovibes::ImageTypeEnum value)
{
    return os << img_type_to_string(value);
}

} // namespace holovibes
