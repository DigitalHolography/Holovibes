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
WindowPixel::WindowPixel(ConversionData data, Axis axis, int val)
    : Unit(data, axis, val)
{
}

WindowPixel::operator OpenglPosition() const
{
    OpenglPosition res(conversion_data_,
                       axis_,
                       conversion_data_.window_size_to_opengl(val_, axis_));
    return res;
}

WindowPixel::operator FDPixel() const
{
    FDPixel res(conversion_data_,
                axis_,
                conversion_data_.opengl_to_fd(
                    static_cast<units::OpenglPosition>(*this).get(),
                    axis_));
    return res;
}
} // namespace units
} // namespace holovibes