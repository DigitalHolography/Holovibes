/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for type of filter 2D
 */
#pragma once

namespace holovibes
{
/*! \brief Type of filter for the Filter2D feature */
enum class Filter2DType
{
    None = 0, /*!<  No filter 2D */
    LowPass,  /*!<  Low pass filter */
    HighPass, /*!<  High pass filter */
    BandPass  /*!<  Band pass filter */
};
} // namespace holovibes