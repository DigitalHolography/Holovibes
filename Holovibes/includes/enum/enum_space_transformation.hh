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

SERIALIZE_JSON_ENUM(SpaceTransformationEnum,
                    {
                        {SpaceTransformationEnum::NONE, "NONE"},
                        {SpaceTransformationEnum::FFT1, "FFT1"},
                        {SpaceTransformationEnum::FFT2, "FFT2"},
                        {SpaceTransformationEnum::FFT1, "1FFT"}, // Compat
                        {SpaceTransformationEnum::FFT2, "2FFT"}, // Compat
                        {SpaceTransformationEnum::NONE, "None"}, // Compat
                    })

inline std::string space_transformation_to_string(SpaceTransformationEnum value)
{
    return _internal::space_transform_to_string.at(value);
}

inline SpaceTransformationEnum space_transformation_from_string(const std::string& in)
{
    return _internal::string_to_space_transform.at(in);
}

inline std::ostream& operator<<(std::ostream& os, SpaceTransformationEnum value)
{
    return os << space_transformation_to_string(value);
}

} // namespace holovibes
