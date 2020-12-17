/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"

namespace holovibes
{
namespace units
{
OpenglPosition::OpenglPosition(ConversionData data, Axis axis, float val)
    : Unit(data, axis, val)
{
}

OpenglPosition::operator FDPixel() const
{
    FDPixel res(conversion_data_,
                axis_,
                conversion_data_.opengl_to_fd(val_, axis_));
    return res;
}

OpenglPosition::operator WindowPixel() const
{
    WindowPixel res(conversion_data_,
                    axis_,
                    conversion_data_.opengl_to_window_size(val_, axis_));
    return res;
}
} // namespace units
} // namespace holovibes
