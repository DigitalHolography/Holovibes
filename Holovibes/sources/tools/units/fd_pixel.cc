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

#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"
#include "real_position.hh"

namespace holovibes
{
	namespace units
	{
		FDPixel::FDPixel(ConversionData data, Axis axis, int val)
			: Unit(data, axis, val)
		{}

		FDPixel::operator OpenglPosition() const
		{
			OpenglPosition res(conversion_data_, axis_, conversion_data_.fd_to_opengl(val_, axis_));
			return res;
		}

		FDPixel::operator WindowPixel() const
		{
			WindowPixel res(conversion_data_, axis_,
				conversion_data_.opengl_to_window_size(static_cast<units::OpenglPosition>(*this).get(), axis_));
			return res;
		}

		FDPixel::operator RealPosition() const
		{
			RealPosition res(conversion_data_, axis_, conversion_data_.fd_to_real(val_, axis_));
			return res;
		}
	}
}
