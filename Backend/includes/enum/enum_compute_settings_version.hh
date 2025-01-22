/*! \file enum_compute_settings_version.hh
 *
 * \brief Enum for the different version of compute settings
 */
#pragma once

#include <map>

#include "all_struct.hh"

namespace holovibes
{
/*! \enum ComputeSettingsVersion
 *
 * \brief All possible versions of compute settings
 */
enum class ComputeSettingsVersion
{
    V2 = 2, /*!< Version 2 */
    V3,     /*!< Version 3 */
    V4,     /*!< Version 4 */
    V5,     /*!< Version 5 */
    V6 /*!< Version 6: First version with the version field inside the compute settings json file. Before it was in the
           header of holo file. */
};

// clang-format off

/* There are that many entries here for the purpose of Retrocompatibility.
 * The original ones were FFT1 and FFT2, but when the names were changed to
 * FRESNELTR and ANGULARSP, they were left here to allow older version of
 * compute settings to work with the new names.
 */
SERIALIZE_JSON_ENUM(ComputeSettingsVersion, {
    {ComputeSettingsVersion::V2, "V2"},
    {ComputeSettingsVersion::V3, "V3"},
    {ComputeSettingsVersion::V4, "V4"},
    {ComputeSettingsVersion::V5, "V5"},
    {ComputeSettingsVersion::V6, "V6"},
})
// clang-format on
} // namespace holovibes
