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
    {SpaceTransformation::NONE, "NONE"},
    {SpaceTransformation::FRESNELTR, "FRESNELTR"}, // Actual saved strings
    {SpaceTransformation::ANGULARSP, "ANGULARSP"},
    {SpaceTransformation::FRESNELTR, "FresnelTR"}, // | (Retro)compatibility
    {SpaceTransformation::ANGULARSP, "AngularSP"}, // |
    {SpaceTransformation::FRESNELTR, "FFT1"},      // |
    {SpaceTransformation::ANGULARSP, "FFT2"},      // |
    {SpaceTransformation::FRESNELTR, "1FFT"},      // |
    {SpaceTransformation::ANGULARSP, "2FFT"},      // |
    {SpaceTransformation::NONE, "None"},           // v
})
// clang-format on
} // namespace holovibes
