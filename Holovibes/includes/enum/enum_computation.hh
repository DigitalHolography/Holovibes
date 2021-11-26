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

namespace _internal
{

const std::map<Computation, std::string> computation_to_string{
    {Computation::Raw, "Raw"},
    {Computation::Hologram, "Hologram"},
};

const std::map<std::string, Computation> string_to_computation{
    {"Raw", Computation::Raw},
    {"Hologram", Computation::Hologram},
};

} // namespace _internal

inline std::string computation_to_string(Computation value) { return _internal::computation_to_string.at(value); }

inline Computation computation_from_string(const std::string& in) { return _internal::string_to_computation.at(in); }

inline std::ostream& operator<<(std::ostream& os, holovibes::Computation value)
{
    return os << computation_to_string(value);
}
} // namespace holovibes