/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *  Enum for the different computation mode
 */
#pragma once

namespace holovibes
{
/*! \brief	Input processes, start at 1 to keep compatibility */
enum class Computation
{
    Raw = 1, /*!< Interferogram recorded */
    Hologram /*!<  Reconstruction of the object */
};
} // namespace holovibes