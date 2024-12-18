/*! \file
 *
 * \brief Enum to differentiate what eye is being recorded
 */
#pragma once

#include <map>
#include <string>
#include "all_struct.hh"

namespace holovibes
{
/*! \enum RecordMode
 *
 * \brief Enum that allows the user to specify what eye is being recorded
 *
 */
enum class RecordedEyeType
{
    LEFT, /*!< The left eye, adds '_L' to the end of the filename */
    NONE, /*!< No eye; this will not affect the filename */
    RIGHT /*!< The right eye, adds '_R' to the end of the filename */
};

// clang-format off
SERIALIZE_JSON_ENUM(RecordedEyeType, {
    {RecordedEyeType::LEFT, "LEFT"},
    {RecordedEyeType::NONE, "NONE"},
    {RecordedEyeType::RIGHT, "RIGHT"}
})

// clang-format on
} // namespace holovibes
