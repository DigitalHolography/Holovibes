/*! \file
 *
 * \brief Enum for the different computation mode
 */
#pragma once

namespace holovibes
{
/*! \enum Computation
 *
 * \brief Input processes, start at 1 to keep compatibility
 */
enum class Computation
{
    Raw = 1, /*!< Interferogram recorded */
    Hologram /*!<  Reconstruction of the object */
};
} // namespace holovibes