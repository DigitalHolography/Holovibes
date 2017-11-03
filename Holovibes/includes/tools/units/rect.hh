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

#pragma once

#include "units/point.hh"
#include "units/window_pixel.hh"
#include "units/fd_pixel.hh"
#include "units/opengl_position.hh"

namespace holovibes
{
	namespace units
	{

		template <typename T>
		class Rect
		{
		public:
			Rect(Point<T> top_left, Point<T> size)
				: top_left_(top_left)
				, size_(size)
			{}

			Rect(ConversionData data,
				typename T::primary_type x0 = 0,
				typename T::primary_type y0 = 0,
				typename T::primary_type x1 = 0,
				typename T::primary_type y1 = 0)
				: top_left_(data, x0, y0)
				, size_(data, x1, y1)
			{}

			Point<T> top_left() const
			{
				return top_left_;
			}

			Point<T> size() const
			{
				return size_;
			}

			Point<T> bottom_right() const
			{
				return top_left_ + size_;
			}

			T x() const
			{
				return top_left_.x();
			}

			T y() const
			{
				return top_left_.y();
			}

			T width() const
			{
				return size.x();
			}

			T height() const
			{
				return size.y();
			}


			template <typename U>
			operator Rect<U>() const
			{
				Rect<U> res(top_left_, size_);
				return res;
			}


		private:
			Point<T> top_left_;
			Point<T> size_;
		};

		using RectOpengl = Rect<OpenglPosition>;
		using RectFd = Rect<FDPixel>;
		using RectWindow = Rect<WindowPixel>;

	}
}

