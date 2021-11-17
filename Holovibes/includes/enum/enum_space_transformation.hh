/*! \file
 *
 * \brief Enum for the different space transformations
 */
#pragma once

namespace holovibes
{
/*! \enum SpaceTransformation
 *
 * \brief Rendering mode for Hologram (Space transformation)
 */
enum class SpaceTransformation
{
    NONE = 0, /*!< Nothing Applied */
    FFT1,     /*!< Fresnel Transform */
    FFT2      /*!< Angular spectrum propagation */
};
} // namespace holovibes