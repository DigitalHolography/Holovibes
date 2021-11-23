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

} // namespace holovibes
