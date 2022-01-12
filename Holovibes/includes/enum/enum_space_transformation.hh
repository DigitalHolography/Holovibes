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

SERIALIZE_JSON_FWD(SpaceTransformation)

// Vestiges, to remove if possible
// these things should pass by the json serializer now
namespace _internal
{

const std::map<std::string, SpaceTransformation> string_to_space_transform = {
    {"NONE", SpaceTransformation::NONE},
    {"None", SpaceTransformation::NONE},
    {"1FFT", SpaceTransformation::FFT1},
    {"2FFT", SpaceTransformation::FFT2},
    {"FFT1", SpaceTransformation::FFT1},
    {"FFT2", SpaceTransformation::FFT2},
};
} // namespace _internal

inline SpaceTransformation space_transformation_from_string(const std::string& in)
{
    return _internal::string_to_space_transform.at(in);
}

} // namespace holovibes
