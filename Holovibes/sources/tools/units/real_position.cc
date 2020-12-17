/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "real_position.hh"
#include "fd_pixel.hh"

namespace holovibes::units
{
RealPosition::RealPosition(ConversionData data, Axis axis, double val)
    : Unit(data, axis, val)
{
}
} // namespace holovibes::units
