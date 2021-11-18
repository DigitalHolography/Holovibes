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
    None = 0, /*!< Nothing Applied */
    FFT1,     /*!< Fresnel Transform */
    FFT2      /*!< Angular spectrum propagation */
};

const std::map<std::string, SpaceTransformation> string_to_space_transform{
    {"None", SpaceTransformation::None}, {"1FFT", SpaceTransformation::FFT1}, {"2FFT", SpaceTransformation::FFT2}};

const std::map<SpaceTransformation, std::string> space_transform_to_string{
    {SpaceTransformation::None, "None"}, {SpaceTransformation::FFT1, "1FFT"}, {SpaceTransformation::FFT2, "2FFT"}};

} // namespace holovibes
