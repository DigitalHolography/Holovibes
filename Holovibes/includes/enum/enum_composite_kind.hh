/*! \file
 *
 * \brief Enum for kind of composite
 */
#pragma once

#include <map>

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

static std::map<CompositeKind, std::string> composite_kind_to_string = {
    {CompositeKind::RGB, "RGB"},
    {CompositeKind::HSV, "HSV"},
};

} // namespace holovibes