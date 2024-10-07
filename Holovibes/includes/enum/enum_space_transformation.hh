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
    FRESNELTR,     /*!< Fresnel Transform */
    ANGULARSP      /*!< Angular spectrum propagation */
};

// clang-format off
SERIALIZE_JSON_ENUM(SpaceTransformation, {
    {SpaceTransformation::NONE, "NONE"},
    {SpaceTransformation::FRESNELTR, "FFT1"}, // Compat
    {SpaceTransformation::ANGULARSP, "FFT2"}, // Compat
    {SpaceTransformation::FRESNELTR, "1FFT"}, // Compat
    {SpaceTransformation::ANGULARSP, "2FFT"}, // Compat
    {SpaceTransformation::FRESNELTR, "FRESNELTR"},
    {SpaceTransformation::ANGULARSP, "ANGULARSP"},
    {SpaceTransformation::FRESNELTR, "FresnelTR"}, // Compat
    {SpaceTransformation::ANGULARSP, "AngularSP"}, // Compat
    {SpaceTransformation::NONE, "None"}, // Compat
})
// clang-format on
} // namespace holovibes
