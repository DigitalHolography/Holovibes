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

#include "Rectangle.hh"
#include <iostream>

namespace holovibes
{
	namespace gui
	{
		#pragma region Constructors
		Rectangle::Rectangle() : QRect()
		{}

		Rectangle::Rectangle(const QRect& rect) : QRect(rect)
		{}

		Rectangle::Rectangle(const Rectangle& rect)
			: QRect()
		{
			setTopLeft(rect.topLeft());
			setBottomRight(rect.bottomRight());
		}

		Rectangle::Rectangle(const QPoint &topleft, const QSize &size)
			: QRect(topleft, size)
		{}

		Rectangle::Rectangle(const uint width, const uint height)
			: QRect(0, 0, width, height)
		{}
		#pragma endregion

		uint	Rectangle::area() const
		{
			return (width() * height());
		}

		void	Rectangle::checkCorners(ushort frameSide, KindOfOverlay kO)
		{
			if (kO == Filter2D)
			{
				if (bottomRight().x() < 0)
					setBottomRight(QPoint(0, bottomRight().y()));
				if (bottomRight().y() < 0)
					setBottomRight(QPoint(bottomRight().x(), 0));

				if (bottomRight().x() > frameSide)
					setBottomRight(QPoint(frameSide, bottomRight().y()));
				if (bottomRight().y() > frameSide)
					setBottomRight(QPoint(bottomRight().x(), frameSide));

				const int length = std::min(std::abs(width()), std::abs(height()));
				setBottomRight(QPoint(
					topLeft().x() +
					length * ((topLeft().x() < bottomRight().x()) * 2 - 1),
					topLeft().y() +
					length * ((topLeft().y() < bottomRight().y()) * 2 - 1)
				));
			}
			if (width() < 0)
			{
				QPoint t0pRight = topRight();
				QPoint b0ttomLeft = bottomLeft();

				setTopLeft(t0pRight);
				setBottomRight(b0ttomLeft);
			}
			if (height() < 0)
			{
				QPoint t0pRight = topRight();
				QPoint b0ttomLeft = bottomLeft();

				setTopLeft(b0ttomLeft);
				setBottomRight(t0pRight);
			}
			// std::cout << *this;
		}

		std::ostream& operator<<(std::ostream& os, const Rectangle& obj)
		{
			os << "topLeft() : " << obj.topLeft().x() << " " << obj.topLeft().y() << std::endl;
			os << "bottomRight() : " << obj.bottomRight().x() << " " << obj.bottomRight().y() << std::endl;
			return (os);
		}

		Rectangle operator-(Rectangle& rec, const QPoint& point)
		{
			Rectangle ret = {0, 0};

			ret.setTopLeft(rec.topLeft() - point);
			ret.setBottomRight(rec.bottomRight() - point);
			return ret;
		}
	}

	std::ostream& operator<<(std::ostream& os, const QPoint& p)
	{
		os << "x: " << p.x() << ", y: " << p.y();
		return os;
	}

	std::ostream& operator<<(std::ostream& os, const QSize& s)
	{
		os << "width: " << s.width() << ", height: " << s.height();
		return os;
	}
}
