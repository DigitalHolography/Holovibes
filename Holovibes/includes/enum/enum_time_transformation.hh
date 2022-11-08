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
    {TimeTransformationEnum::NONE, "None"}, // Compat

});
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const TimeTransformationEnum& value) { return os << json{value}; }

} // namespace holovibes
