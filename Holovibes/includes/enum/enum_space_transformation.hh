/*! \file
 *
 * \brief Enum for the different space transformations
 */
#pragma once

#include <map>

#include "all_struct.hh"

namespace holovibes
{
/*! \enum SpaceTransformationEnum
 *
 * \brief Rendering mode for Hologram (Space transformation)
 */
enum class SpaceTransformationEnum
{
    NONE = 0, /*!< Nothing Applied */
    FFT1,     /*!< Fresnel Transform */
    FFT2      /*!< Angular spectrum propagation */
};

// clang-format off
SERIALIZE_JSON_ENUM(SpaceTransformationEnum,
{
    {SpaceTransformationEnum::NONE, "NONE"},
    {SpaceTransformationEnum::FFT1, "FFT1"},
    {SpaceTransformationEnum::FFT2, "FFT2"},
    {SpaceTransformationEnum::FFT1, "1FFT"}, // Compat
    {SpaceTransformationEnum::FFT2, "2FFT"}, // Compat
    {SpaceTransformationEnum::NONE, "None"}, // Compat
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const SpaceTransformationEnum& value) { return os << json{value}; }

} // namespace holovibes
