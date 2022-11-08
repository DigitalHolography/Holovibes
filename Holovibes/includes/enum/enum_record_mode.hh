/*! \file
 *
 * \brief Enum for the different type of record
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum RecordMode
 *
 * \brief #TODO Add a description for this enum
 */
enum class RecordMode
{
    NONE,
    CHART,
    CUTS_XZ,
    CUTS_YZ,
    HOLOGRAM,
    RAW,
};

// clang-format off
SERIALIZE_JSON_ENUM(RecordMode, {
    {RecordMode::NONE, "NONE"},
    {RecordMode::CHART, "CHART"},
    {RecordMode::CUTS_XZ, "CUTS_XZ"},
    {RecordMode::CUTS_YZ, "CUTS_YZ"},
    {RecordMode::HOLOGRAM, "HOLOGRAM"},
    {RecordMode::RAW, "RAW"}
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const RecordMode& value) { return os << json{value}; }
} // namespace holovibes