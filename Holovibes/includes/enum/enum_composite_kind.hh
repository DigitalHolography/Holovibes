/*! \file
 *
 * \brief Enum for kind of composite
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum CompositeKindEnum
 *
 * \brief Represents the kind of composite image
 */
enum class CompositeKindEnum
{
    RGB = 0, /*!< Composite in RGB */
    HSV      /*!< Composite in HSV */
};

// clang-format off
SERIALIZE_JSON_ENUM(CompositeKindEnum, {
    {CompositeKindEnum::RGB, "RGB"},
    {CompositeKindEnum::HSV, "HSV"},
})
// clang-format on

inline std::ostream& operator<<(std::ostream& os, const CompositeKindEnum& value) { return os << json{value}; }

} // namespace holovibes