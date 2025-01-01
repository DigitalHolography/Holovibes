#include "fd_pixel.hh"
#include "opengl_position.hh"

namespace holovibes::units
{
FDPixel::FDPixel(ConversionData data, Axis axis, int val)
    : Unit(data, axis, val)
{
}

FDPixel::operator OpenglPosition() const
{
    OpenglPosition res(conversion_data_, axis_, conversion_data_.fd_to_opengl(val_, axis_));
    return res;
}
} // namespace holovibes::units
