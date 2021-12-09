/*! \file
 *
 * \brief Enum for the different time transformations
 */
#pragma once

#include <map>
#include <ostream>

namespace holovibes
{
/*! \enum TimeTransformation
 *
 * \brief Time transformation algorithm to apply
 */
enum class TimeTransformation
{
    STFT = 0, /*!< Short-time Fourier transformation */
    PCA,      /*!< Principal component analysis */
    NONE,     /*!< No transformation */
    SSA_STFT  /*!< Self-adaptive Spectrum Analysis Short-time Fourier transformation */
};

namespace _internal
{

const std::map<TimeTransformation, std::string> time_transformation_to_string = {
    {TimeTransformation::STFT, "STFT"},
    {TimeTransformation::PCA, "PCA"},
    {TimeTransformation::NONE, "NONE"},
    {TimeTransformation::SSA_STFT, "SSA_STFT"},
};

const std::map<std::string, TimeTransformation> string_to_time_transformation = {
    {"STFT", TimeTransformation::STFT},
    {"PCA", TimeTransformation::PCA},
    {"NONE", TimeTransformation::NONE},
    {"SSA_STFT", TimeTransformation::SSA_STFT},
};

} // namespace _internal

inline std::string time_transformation_to_string(TimeTransformation value)
{
    return _internal::time_transform_to_string.at(value);
}

inline TimeTransformation time_transformation_from_string(const std::string& in)
{
    return _internal::string_to_time_transform.at(in);
}

inline std::ostream& operator<<(std::ostream& os, holovibes::TimeTransformation value)
{
    return os << time_transformation_to_string(value);
}

} // namespace holovibes
