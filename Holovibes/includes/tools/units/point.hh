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

/*! \file
 *
 * Implementation of a Point */
#pragma once

#include "conversion_data.hh"
#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"
#include "real_position.hh"

#include <type_traits>


namespace holovibes
{
	namespace units
	{

		/*! \brief A point in some specific unit
		 */
		template <class T>
		class Point
		{
		public:
			Point()
				: x_(ConversionData(), Axis::HORIZONTAL, 0)
				, y_(ConversionData(), Axis::VERTICAL, 0)
			{}

			Point(T x, T y)
				: x_(x)
				, y_(y)
			{}

			/*! \brief Constructs a point from the needed conversion data and two primary types
			 */
			Point(ConversionData data, typename T::primary_type x = 0, typename T::primary_type y = 0)
				: x_(data, Axis::HORIZONTAL, x)
				, y_(data, Axis::VERTICAL, y)
			{}

			T& x()
			{
				return x_;
			}

			T& y()
			{
				return y_;
			}

			T x() const
			{
				return x_;
			}

			T y() const
			{
				return y_;
			}

			/*! \brief Implicit cast into a point of an other unit
			 */
			template <typename U>
			operator Point<U>() const
			{
				Point<OpenglPosition> tmp(x_, y_);
				// We can't use "if constexpr" here because cuda isn't c++17
				// Once it is, please add the constexpr
				if (std::is_same<T, FDPixel>::value)
					x_.getConversion().transform_from_fd(tmp.x(), tmp.y());
				if (std::is_same<U, FDPixel>::value)
					x_.getConversion().transform_to_fd(tmp.x(), tmp.y());
				Point<U> res(tmp.x(), tmp.y());
				return res;
			}

			// TODO: fix the warning saying that it won't be called.
			operator Point<RealPosition>() const
			{
				Point<RealPosition> res(x_, y_);
				return res;
			}
			/*! \brief Operator overloads
			 */
			/**@{*/
			Point<T> operator+(const Point<T>& other) const
			{
				Point<T> res(x_, y_);
				res.x_ += other.x_;
				res.y_ += other.y_;
				return res;
			}

			Point<T> operator-(const Point<T>& other) const
			{
				Point<T> res(x_, y_);
				res.x_ -= other.x_;
				res.y_ -= other.y_;
				return res;
			}

			double distance() const
			{
				return sqrt(pow(x_, 2) + pow(y_, 2));
			}

			bool operator==(const Point<T>& other) const
			{
				return x_ == other.x_ && y_ == other.y_;
			}
			/**@}*/

		private:
			T x_;
			T y_;
		};

		/*! \brief A point in the OpenGL coordinate system [-1;1]
		 */
		using PointOpengl = Point<OpenglPosition>;

		/*! \brief A point in the frame desc coordinate system [0;fd.width]
		 */
		using PointFd = Point<FDPixel>;

		/*! \brief A point in the window coordinate system [0;window size]
		 */
		using PointWindow = Point<WindowPixel>;

		using PointReal= Point<RealPosition>;

		template<typename T>
		std::ostream& operator<<(std::ostream& o, const Point<T>& p)
		{
			return o << '(' << p.x() << ", " << p.y() << ')';
		}

	}
}

