/*! \file
 *
 * \brief Enum for the different computation mode
 */
#pragma once

#include <map>
#include "all_struct.hh"

namespace holovibes
{
/*! \enum Computation
 *
 * \brief Input processes
 */
enum class Computation
{
    Raw = 0, /*!< Interferogram recorded */
    Hologram /*!<  Reconstruction of the object */
};

// clang-format off
SERIALIZE_JSON_ENUM(Computation, {
    {Computation::Raw, "RAW"},
    {Computation::Hologram, "HOLOGRAM"},
})
// clang-format on

} // namespace holovibes
