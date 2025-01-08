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
 * This has no influence on computations, this only modifies a recording file's name
 *
 */
enum class RecordedEyeType
{
    LEFT = 0, /*!< The left eye, adds '_L' to the end of the filename */
    RIGHT,    /*!< The right eye, adds '_R' to the end of the filename */
    NONE      /*!< No eye; this will not affect the filename */
};

// clang-format off
SERIALIZE_JSON_ENUM(RecordedEyeType, {
    {RecordedEyeType::LEFT, "LEFT"},
    {RecordedEyeType::RIGHT, "RIGHT"},
    {RecordedEyeType::NONE, "NONE"}
})

// clang-format on
} // namespace holovibes
