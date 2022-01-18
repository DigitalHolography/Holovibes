/*! \file
 *
 * \brief Enum for the different space transformations
 */
#pragma once

#include <map>

#include "all_struct.hh"

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

// clang-format off
SERIALIZE_JSON_ENUM(SpaceTransformation, {
    {SpaceTransformation::NONE, "NONE"},
    {SpaceTransformation::FFT1, "FFT1"},
    {SpaceTransformation::FFT2, "FFT2"},
    {SpaceTransformation::FFT1, "1FFT"}, // Compat
    {SpaceTransformation::FFT2, "2FFT"}, // Compat
})
// clang-format on
} // namespace holovibes
