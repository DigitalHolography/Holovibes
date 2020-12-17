
/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for the different type of displaying images
 */
#pragma once

namespace holovibes
{
/*! \brief	Displaying type of the image */
enum class ImgType
{
    Modulus = 0,    /*!< Modulus of the complex data */
    SquaredModulus, /*!<  Modulus taken to its square value */
    Argument, /*!<  Phase (angle) value of the complex pixel c, computed with
                 atan(Im(c)/Re(c)) */
    PhaseIncrease, /*!<  Phase value computed with the conjugate between the
                      phase of the last image and the previous one */
    Composite /*!<  Displays different frequency intervals on color RBG or HSV
                 chanels*/
};
} // namespace holovibes