/*! \file
 *
 * \brief Enum for the different space transformations
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