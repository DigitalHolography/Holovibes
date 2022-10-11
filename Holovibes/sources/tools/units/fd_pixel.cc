#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"
#include "real_position.hh"

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

FDPixel::operator WindowPixel() const
{
    WindowPixel res(conversion_data_,
                    axis_,
                    conversion_data_.opengl_to_window_size(static_cast<units::OpenglPosition>(*this).get(), axis_));
    return res;
}

FDPixel::operator RealPosition() const
{
    RealPosition res(conversion_data_, axis_, conversion_data_.fd_to_real(val_, axis_));
    return res;
}
} // namespace holovibes::units
