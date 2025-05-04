/*! \file
 *
 * \brief Enum for the two otsu algorithms
 */
#pragma once

#include "all_struct.hh"

namespace holovibes
{
/*! \enum OtsuKind
 *
 * \brief Which Otsu algorithm is used
 */
enum class OtsuKind
{
    Global = 0, // Base otsu, simple
    Adaptive    // Uses more parameters to refine the algorithm
};

// clang-format off
SERIALIZE_JSON_ENUM(OtsuKind, {
    {OtsuKind::Global, "Global"},
    {OtsuKind::Adaptive, "Adaptive"}
})
// clang-format on

} // namespace holovibes
