#include "geometry.hh"

namespace holovibes
{
  /*! Point in 2D */

  Point2D::Point2D()
  {
    x = 0;
    y = 0;
  }

  Point2D::Point2D(const Point2D& p)
  {
    x = p.x;
    y = p.y;
  }

  Point2D::Point2D(const int xcoord, const int ycoord)
  {
    x = xcoord;
    y = ycoord;
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

  Rectangle::Rectangle()
  {
  }

  Rectangle::Rectangle(const Point2D top_left_corner, const Point2D bottom_right_corner)
  {
    top_left = top_left_corner;
    bottom_right = bottom_right_corner;
    top_right = Point2D(bottom_right_corner.x, top_left_corner.y);
    bottom_left = Point2D(top_left_corner.x, bottom_right_corner.y);
  }

  Rectangle::Rectangle(const Rectangle& rect)
  {
    top_left = rect.top_left;
    top_right = rect.top_right;
    bottom_left = rect.bottom_left;
    bottom_right = rect.bottom_right;
  }

  Rectangle& Rectangle::operator=(const Rectangle& rect)
  {
    top_left = rect.top_left;
    top_right = rect.top_right;
    bottom_left = rect.bottom_left;
    bottom_right = rect.bottom_right;
    return *this;
  }

  unsigned int Rectangle::area() const
  {
    const unsigned int AB = bottom_right.x - top_left.x;
    const unsigned int AD = bottom_right.y - top_left.y;
    return (AB * AD);
  }

  unsigned int Rectangle::get_width() const
  {
    const unsigned int AB = bottom_right.x - top_left.x;
    return (AB);
  }

  unsigned int Rectangle::get_height() const
  {
    const unsigned int AD = bottom_right.y - top_left.y;
    return (AD);
  }

  void Rectangle::vertical_symetry()
  {
    Point2D tmp;
    tmp = top_left;
    top_left = top_right;
    top_right = tmp;

    tmp = bottom_left;
    bottom_left = bottom_right;
    bottom_right = tmp;
  }

  void Rectangle::horizontal_symetry()
  {
    Point2D tmp;
    tmp = top_left;
    top_left = bottom_left;
    bottom_left = tmp;

    tmp = top_right;
    top_right = bottom_right;
    bottom_right = tmp;
  }
}