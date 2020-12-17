/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for the different space transformations
 */
#pragma once

namespace holovibes
{
/*! \brief	Rendering mode for Hologram (Space transformation) */
enum class SpaceTransformation
{
    None = 0, /*!< Nothing Applied */
    FFT1,     /*!< Fresnel Transform */
    FFT2      /*!< Angular spectrum propagation */
};
} // namespace holovibes