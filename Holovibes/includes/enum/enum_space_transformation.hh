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
    NONE = 0,  /*!< Nothing Applied */
    FRESNELTR, /*!< Fresnel Transform */
    ANGULARSP  /*!< Angular spectrum propagation */
};

// clang-format off

/* There are that many entries here for the purpose of Retrocompatibility.
 * The original ones were FFT1 and FFT2, but when the names were changed to
 * FRESNELTR and ANGULARSP, they were left here to allow older version of
 * compute settings to work with the new names.
 */
SERIALIZE_JSON_ENUM(SpaceTransformation, {
    {SpaceTransformation::NONE, "NONE"},           // Actual saved name
    {SpaceTransformation::NONE, "None"},           //   (Retro)compatibility
    {SpaceTransformation::FRESNELTR, "FRESNELTR"}, // Actual saved name
    {SpaceTransformation::FRESNELTR, "FresnelTR"}, // | (Retro)compatibility
    {SpaceTransformation::FRESNELTR, "FFT1"},      // |
    {SpaceTransformation::FRESNELTR, "1FFT"},      // v
    {SpaceTransformation::ANGULARSP, "ANGULARSP"}, // Actual saved name
    {SpaceTransformation::ANGULARSP, "AngularSP"}, // | (Retro)compatibility
    {SpaceTransformation::ANGULARSP, "FFT2"},      // |
    {SpaceTransformation::ANGULARSP, "2FFT"},      // v
})
// clang-format on
} // namespace holovibes
