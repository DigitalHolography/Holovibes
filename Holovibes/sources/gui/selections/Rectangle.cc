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
			if (kO == Filter2D)
			{
				if (topRight().x() > frameSide)
				{
					setTopRight(QPoint(frameSide, topRight().y()));

					const int min = std::min(width(), height());
					setBottomRight(QPoint(
						topLeft().x() +
						min * ((topLeft().x() < bottomRight().x()) * 2 - 1),
						topLeft().y() +
						min * ((topLeft().y() < bottomRight().y()) * 2 - 1)
					));
				}
				if (bottomRight().y() > frameSide)
				{
					setBottomRight(QPoint(bottomRight().x(), frameSide));
					const int min = std::min(width(), height());
					setBottomRight(QPoint(
						topLeft().x() +
						min * ((topLeft().x() < bottomRight().x()) * 2 - 1),
						topLeft().y() +
						min * ((topLeft().y() < bottomRight().y()) * 2 - 1)
					));
				}
			}
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
