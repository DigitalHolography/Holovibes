/*! \file
 *
 * \brief Enum for the different ComputeModeEnum mode
 */
#pragma once

#include <map>
#include "all_struct.hh"

namespace holovibes
{
/*! \enum ComputeModeEnum
 *
 * \brief Input processes
 */
enum class ComputeModeEnum
{
    Raw = 0, /*!< Interferogram recorded */
    Hologram /*!<  Reconstruction of the object */
};

// clang-format off
SERIALIZE_JSON_ENUM(ComputeModeEnum, {
    {ComputeModeEnum::Raw, "RAW"},
    {ComputeModeEnum::Hologram, "HOLOGRAM"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const ComputeModeEnum& value) { return os << json{value}; }

} // namespace holovibes