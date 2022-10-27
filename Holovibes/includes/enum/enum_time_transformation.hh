/*! \file
 *
 * \brief Enum for the different time transformations
 */
#pragma once

#include <map>
#include "all_struct.hh"

namespace holovibes
{
/*! \enum TimeTransformationEnum
 *
 * \brief Time transformation algorithm to apply
 */
enum class TimeTransformationEnum
{
    STFT = 0, /*!< Short-time Fourier transformation */
    PCA,      /*!< Principal component analysis */
    NONE,     /*!< No transformation */
    SSA_STFT  /*!< Self-adaptive Spectrum Analysis Short-time Fourier transformation */
};

// clang-format off
SERIALIZE_JSON_ENUM(TimeTransformationEnum, {
    {TimeTransformationEnum::STFT, "STFT"},
    {TimeTransformationEnum::PCA, "PCA"},
    {TimeTransformationEnum::NONE, "NONE"},
    {TimeTransformationEnum::SSA_STFT, "SSA_STFT"},
    {TimeTransformation::NONE, "None"}, // Compat

});

inline std::string time_transformation_to_string(TimeTransformationEnum value)
{
    return _internal::time_transform_to_string.at(value);
}

inline TimeTransformationEnum time_transformation_from_string(const std::string& in)
{
    return _internal::string_to_time_transform.at(in);
}

inline std::ostream& operator<<(std::ostream& os, holovibes::TimeTransformationEnum value)
{
    return os << time_transformation_to_string(value);
}

} // namespace holovibes
