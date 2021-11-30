/*! \file
 *
 * \brief Enum for the different computation mode
 */
#pragma once

#include <map>

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

static std::map<std::string, Computation> string_to_computation = {
    {"RAW", Computation::Raw},
    {"HOLOGRAM", Computation::Hologram},
};

static std::map<Computation, std::string> computation_to_string = {
    {Computation::Raw, "RAW"},
    {Computation::Hologram, "HOLOGRAM"},
};

} // namespace holovibes