/*! \file
 *
 * \brief Enum for the different time transformations
 */
#pragma once

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

#define MAP_AND_REVERSE(type1, type2, ...)

const std::map<TimeTransformation, std::string> time_transform_to_string {
    {TimeTransformation::STFT, "STFT",},
    {TimeTransformation::PCA, "PCA",},
    {TimeTransformation::NONE, "None",},
    {TimeTransformation::SSA_STFT, "SSA_STFT",}
};

const std::map<std::string, TimeTransformation> string_to_time_transform{
    {"STFT", TimeTransformation::STFT},
    {"PCA", TimeTransformation::PCA},
    {"None", TimeTransformation::NONE},
    {"SSA_STFT", TimeTransformation::SSA_STFT}
};

} // namespace holovibes
