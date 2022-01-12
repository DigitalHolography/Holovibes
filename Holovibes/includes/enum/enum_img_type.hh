
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
    Composite       /*!<  Displays different frequency intervals on color RBG or HSV chanels*/
};
} // namespace holovibes

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(ImgType)

} // namespace holovibes
