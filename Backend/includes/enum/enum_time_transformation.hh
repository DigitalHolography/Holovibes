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
    NONE = 0, /*!< No transformation */
    STFT,     /*!< Short-time Fourier transformation */
    PCA,      /*!< Principal component analysis */
    SSA_STFT, /*!< Self-adaptive Spectrum Analysis Short-time Fourier transformation */
    STFT_SSA, /*!< Short-time Fourier transformation Self-adaptive Spectrum Analysis */
    WAVELET   /*!< Wavelet Transform */
};

// clang-format off
SERIALIZE_JSON_ENUM(TimeTransformation, {
    {TimeTransformation::STFT, "STFT"},
    {TimeTransformation::PCA, "PCA"},
    {TimeTransformation::NONE, "NONE"},
    {TimeTransformation::SSA_STFT, "SSA+STFT"},
    {TimeTransformation::STFT_SSA, "STFT+SSA"},
    {TimeTransformation::WAVELET, "WAVELET"},
    {TimeTransformation::NONE, "None"}, // Compat

})
// clang-format on
} // namespace holovibes
