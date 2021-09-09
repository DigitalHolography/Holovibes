/*! \file
 *  Enum for the different time transformations
 */
#pragma once

namespace holovibes
{
/*! \brief	Time transformation algorithm to apply */
enum class TimeTransformation
{
    STFT = 0, /*!< Short-time Fourier transformation */
    PCA,      /*!< Principal component analysis */
    NONE,
    SSA_STFT
};
} // namespace holovibes
