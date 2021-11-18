/*! \file
 *
 * \brief Enum for the different computation mode
 */
#pragma once

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
} // namespace holovibes