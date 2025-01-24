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
    NONE = 0, /*!< No version, for old compute settings that does not have a version */
    V1, /*!< Version 1: introduce version, image registration and deletion of the enabled field for convolution and
           input filter */
};

// clang-format off

/* There are that many entries here for the purpose of Retrocompatibility.
 * The original ones were FFT1 and FFT2, but when the names were changed to
 * FRESNELTR and ANGULARSP, they were left here to allow older version of
 * compute settings to work with the new names.
 */
SERIALIZE_JSON_ENUM(ComputeSettingsVersion, {
    {ComputeSettingsVersion::NONE, "None"},
    {ComputeSettingsVersion::NONE, "v0"},
    {ComputeSettingsVersion::V1, "v1"},
})
// clang-format on
} // namespace holovibes
