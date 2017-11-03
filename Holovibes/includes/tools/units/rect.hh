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
			 * \param size Size of the rectangle, x = width, y = height
			 */
			Rect(Point<T> top_left, Point<T> size)
				: top_left_(top_left)
				, size_(size)
			{}

			/*! \brief Constructs a rectangle from its position and size
			 */
			Rect(ConversionData data,
				typename T::primary_type x = 0,
				typename T::primary_type y = 0,
				typename T::primary_type width = 0,
				typename T::primary_type height = 0)
				: top_left_(data, x, y)
				, size_(data, width, height)
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


			/*! \brief Implicit cast into a rectangle of an other unit
			 */
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

