/*! \file
 *
 * \brief Enum for the different time transformations
 */
#pragma once

#include <map>
#include "all_struct.hh"

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

// clang-format off
SERIALIZE_JSON_ENUM(TimeTransformation, {
    {TimeTransformation::STFT, "STFT"},
    {TimeTransformation::PCA, "PCA"},
    {TimeTransformation::NONE, "NONE"},
    {TimeTransformation::SSA_STFT, "SSA_STFT"},
    {TimeTransformation::NONE, "None"}, // Compat

})
// clang-format on
} // namespace holovibes
