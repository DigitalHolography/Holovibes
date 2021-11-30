/*! \file
 *
 * \brief Enum for the different time transformations
 */
#pragma once

#include <map>

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

static std::map<TimeTransformation, std::string> time_transformation_to_string = {
    {TimeTransformation::STFT, "STFT"},
    {TimeTransformation::PCA, "PCA"},
    {TimeTransformation::NONE, "NONE"},
    {TimeTransformation::SSA_STFT, "SSA_STFT"},
};

static std::map<std::string, TimeTransformation> string_to_time_transformation = {
    {"STFT", TimeTransformation::STFT},
    {"PCA", TimeTransformation::PCA},
    {"NONE", TimeTransformation::NONE},
    {"SSA_STFT", TimeTransformation::SSA_STFT},
};

} // namespace holovibes
