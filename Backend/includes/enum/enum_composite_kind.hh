/*! \file
 *
 * \brief Enum for kind of composite
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum CompositeKind
 *
 * \brief Represents the kind of composite image
 */
enum class CompositeKind
{
    RGB = 0, /*!< Composite in RGB */
    HSV      /*!< Composite in HSV */
};

// clang-format off
SERIALIZE_JSON_ENUM(CompositeKind, {
    {CompositeKind::RGB, "RGB"},
    {CompositeKind::HSV, "HSV"},
})
// clang-format on

} // namespace holovibes
