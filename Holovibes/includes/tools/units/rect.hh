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

		/*! \brief A rectangle in some specific unit
		 */
		template <typename T>
		class Rect
		{
		public:
			/*! \brief Constructs a rectangle from two points
			 * 
			 * \param top_left Top left point
			 * \param size bottom_right Bottom right point
			 */
			Rect(Point<T> top_left, Point<T> bottom_right)
				: top_left_(top_left)
				, bottom_right_(bottom_right)
			{}

			/*! \brief Constructs a rectangle from its position and size
			 */
			Rect(ConversionData data,
				typename T::primary_type x1 = 0,
				typename T::primary_type y1 = 0,
				typename T::primary_type x2 = 0,
				typename T::primary_type y2 = 0)
				: top_left_(data, x1, y1)
				, bottom_right_(data, x2, y2)
			{}

			const Point<T>& top_left() const
			{
				return top_left_;
			}

			const Point<T>& bottom_right() const
			{
				return bottom_right_;
			}

			Point<T> size() const
			{
				return bottom_right_ - top_left_;
			}

			T x() const
			{
				return top_left_.x();
			}

			T& x()
			{
				return top_left_.x();
			}

			T y() const
			{
				return top_left_.y();
			}

			T& y()
			{
				return top_left_.y();
			}

			T width() const
			{
				return size().x();
			}

			T height() const
			{
				return size().y();
			}


			/*! \brief Implicit cast into a rectangle of an other unit
			 */
			template <typename U>
			operator Rect<U>() const
			{
				Rect<U> res(top_left_, bottom_right_);
				return res;
			}


		private:
			Point<T> top_left_;
			Point<T> bottom_right_;
		};

		/*! \brief Rectangle in the OpenGL coordinates [-1;1]
		 */
		using RectOpengl = Rect<OpenglPosition>;

		/*! \brief Rectangle in the frame desc coordinates
		 */
		using RectFd = Rect<FDPixel>;

		/*! \brief Rectangle in the window coordinates
		 */
		using RectWindow = Rect<WindowPixel>;

	}
}

