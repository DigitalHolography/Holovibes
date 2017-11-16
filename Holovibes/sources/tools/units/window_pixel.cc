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

#include "units/window_pixel.hh"
#include "units/fd_pixel.hh"
#include "units/opengl_position.hh"

#include <iostream>

namespace holovibes
{
	namespace units
	{
		WindowPixel::WindowPixel(ConversionData data, Axis axis, int val)
			: Unit(data, axis, val)
		{}

		WindowPixel::operator OpenglPosition() const
		{
			OpenglPosition res(conversion_data_, axis_, conversion_data_.window_size_to_opengl(val_, axis_));
			return res;
		}

		WindowPixel::operator FDPixel() const
		{
			FDPixel res(conversion_data_, axis_,
					conversion_data_.opengl_to_fd(static_cast<units::OpenglPosition>(*this).get(), axis_));
			return res;
		}
	}
}