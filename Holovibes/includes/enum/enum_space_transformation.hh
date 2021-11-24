/*! \file
 *
 * \brief Enum for the different space transformations
 */
#pragma once

#include <map>

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

static std::map<SpaceTransformation, std::string> space_transformation_to_string = {
    {SpaceTransformation::NONE, "NONE"},
    {SpaceTransformation::FFT1, "FFT1"},
    {SpaceTransformation::FFT2, "FFT2"},
};

static std::map<std::string, SpaceTransformation> string_to_space_transformation = {
    {"NONE", SpaceTransformation::NONE},
    {"FFT1", SpaceTransformation::FFT1},
    {"FFT2", SpaceTransformation::FFT2},
};

} // namespace holovibes