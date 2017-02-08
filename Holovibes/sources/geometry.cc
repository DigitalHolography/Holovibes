#include "geometry.hh"

namespace holovibes
{
	#pragma region 2D Point
	Point2D::Point2D()
		: x(0)
		, y(0)
	{
	}

	Point2D::Point2D(const Point2D& p)
		: x(p.x)
		, y(p.y)
	{
	}

	Point2D::Point2D(const int xcoord, const int ycoord)
		: x(xcoord)
		, y(ycoord)
	{
	}

	Point2D& Point2D::operator=(const Point2D& p)
	{
		x = p.x;
		y = p.y;
		return *this;
	}

	bool Point2D::operator!=(const Point2D& p)
	{
		return x != p.x || y != p.y;
	}
	#pragma endregion

	#pragma region Rectangle
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
	#pragma endregion

	/*void Rectangle::vertical_symetry()
	{
		Point2D tmp(top_left);
		top_left = top_right;
		top_right = tmp;

		tmp = bottom_left;
		bottom_left = bottom_right;
		bottom_right = tmp;
	}

	void Rectangle::horizontal_symetry()
	{
		Point2D tmp(top_left);
		top_left = bottom_left;
		bottom_left = tmp;

		tmp = top_right;
		top_right = bottom_right;
		bottom_right = tmp;
	}*/
}