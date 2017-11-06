/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "units\conversion_data.hh"
#include "units\unit.hh"

using holovibes::units::ConversionData;
using holovibes::units::Axis;

ConversionData::ConversionData(const BasicOpenGLWindow& window)
	: window_(window)
{}

float ConversionData::window_size_to_opengl(int val, Axis axis) const
{
	return (static_cast<float>(val) * 2.f / static_cast<float>(get_window_size(axis))) - 1;
}

float ConversionData::fd_to_opengl(int val, Axis axis) const
{
	return (static_cast<float>(val) * 2.f / static_cast<float>(get_fd_size(axis))) - 1;
}

int ConversionData::opengl_to_window_size(float val, Axis axis) const
{
	return ((val + 1.f) / 2.f) * get_window_size(axis);
}

int ConversionData::opengl_to_fd(float val, Axis axis) const
{
	return ((val + 1.f) / 2.f) * get_fd_size(axis);
}

int ConversionData::get_window_size(Axis axis) const
{
	switch (axis)
	{
	case Axis::HORIZONTAL:
		return window_.width();
	case Axis::VERTICAL:
		return window_.height();
	default:
		throw std::exception("Unreachable code");
	}
}

int ConversionData::get_fd_size(Axis axis) const
{
	switch (axis)
	{
	case Axis::HORIZONTAL:
		return window_.getFd().width;
	case Axis::VERTICAL:
		return window_.getFd().height;
	default:
		throw std::exception("Unreachable code");
	}
}
