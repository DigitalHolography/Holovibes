#include "geometry.hh"

namespace holovibes
{
	Rectangle::Rectangle() : QRect()
	{
	}

	Rectangle::Rectangle(const Rectangle& rect)
		: QRect()
	{
		setTopRight(rect.topRight());
		setTopLeft(rect.topLeft());
		setBottomRight(rect.bottomRight());
		setBottomLeft(rect.bottomLeft());
	}

	Rectangle::Rectangle(const QPoint &topleft, const QSize &size)
		: QRect(topleft, size)
	{
	}

	Rectangle::Rectangle(const unsigned int width, const unsigned int height)
		: QRect(0, 0, width, height)
	{
	}

	unsigned int Rectangle::area() const
	{
		return (width() * height());
	}
}