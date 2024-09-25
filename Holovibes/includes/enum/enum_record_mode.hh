/*! \file
 *
 * \brief Enum for the different type of record
 */
#pragma once

#include <map>
#include <string>
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
    MOMENTS,
};

// clang-format off
SERIALIZE_JSON_ENUM(RecordMode, {
    {RecordMode::NONE, "NONE"},
    {RecordMode::CUTS_YZ, "CUTS_YZ"},
    {RecordMode::RAW, "RAW"},
    {RecordMode::HOLOGRAM, "HOLOGRAM"},
    {RecordMode::CHART, "CHART"},
    {RecordMode::CUTS_XZ, "CUTS_XZ"},
    {RecordMode::MOMENTS, "MOMENTS"}
})
// clang-format on
} // namespace holovibes