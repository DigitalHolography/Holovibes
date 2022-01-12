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
} // namespace holovibes

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(TimeTransformation)

// Vestiges, to remove if possible
// these things should pass by the json serializer now
namespace _internal
{

const std::map<std::string, TimeTransformation> string_to_time_transform = {
    {"STFT", TimeTransformation::STFT},
    {"PCA", TimeTransformation::PCA},
    {"NONE", TimeTransformation::NONE},
    {"None", TimeTransformation::NONE},
    {"SSA_STFT", TimeTransformation::SSA_STFT},
};

} // namespace holovibes::_internal

inline TimeTransformation time_transformation_from_string(const std::string& in)
{
    return _internal::string_to_time_transform.at(in);
}

} // namespace holovibes
