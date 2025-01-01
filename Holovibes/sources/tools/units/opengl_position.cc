#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"

namespace holovibes::units
{
OpenglPosition::OpenglPosition(ConversionData data, Axis axis, float val)
    : Unit(data, axis, val)
{
}

OpenglPosition::operator FDPixel() const
{
    FDPixel res(conversion_data_, axis_, conversion_data_.opengl_to_fd(val_, axis_));
    return res;
}
} // namespace holovibes::units
