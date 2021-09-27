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
} // namespace holovibes
