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
} // namespace holovibes

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_FWD(CompositeKind)

} // namespace holovibes
