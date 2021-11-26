
/*! \file
 *
 * \brief Enum for the different type of displaying images
 */
#pragma once

#include <map>

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
    Composite       /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
};

static std::map<std::string, ImgType> string_to_img_type = {
    {"MODULUS", ImgType::Modulus},
    {"SQUAREMODULUS", ImgType::SquaredModulus},
    {"ARGUMENT", ImgType::Argument},
    {"PHASEINCREASE", ImgType::PhaseIncrease},
    {"COMPOSITE", ImgType::Composite},
};

static std::map<ImgType, std::string> img_type_to_string = {
    {ImgType::Modulus, "MODULUS"},
    {ImgType::SquaredModulus, "SQUAREDMODULUS"},
    {ImgType::Argument, "ARGUMENT"},
    {ImgType::PhaseIncrease, "PHASEINCREASE"},
    {ImgType::Composite, "COMPOSITE"},
};

} // namespace holovibes