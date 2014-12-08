#ifndef GEOMETRY_HH
# define GEOMETRY_HH

namespace holovibes
{
  struct Point2D
  {
    int x;
    int y;

    Point2D()
    {
      x = 0;
      y = 0;
    }

    Point2D(const Point2D& p)
    {
      x = p.x;
      y = p.y;
    }

    Point2D(int xcoord, int ycoord)
    {
      x = xcoord;
      y = ycoord;
    }

    Point2D& operator=(const Point2D& p)
    {
      x = p.x;
      y = p.y;
      return *this;
    }
  };

  struct Rectangle
  {
    Point2D top_left;
    Point2D top_right;
    Point2D bottom_left;
    Point2D bottom_right;

    Rectangle()
    {
    }

    Rectangle(Point2D top_left_corner, Point2D bottom_right_corner)
    {
      top_left = top_left_corner;
      bottom_right = bottom_right_corner;
      top_right = Point2D(bottom_right_corner.x, top_left_corner.y);
      bottom_left = Point2D(top_left_corner.x, bottom_right_corner.y);
    }

    Rectangle(Point2D top_left_corner,
      Point2D top_right_corner,
      Point2D bottom_left_corner,
      Point2D bottom_right_corner)
    {
      top_left = top_left_corner;
      top_right = top_right_corner;
      bottom_left = bottom_left_corner;
      bottom_right = bottom_right_corner;
    }

    Rectangle(const Rectangle& rect)
    {
      top_left = rect.top_left;
      top_right = rect.top_right;
      bottom_left = rect.bottom_left;
      bottom_right = rect.bottom_right;
    }

    Rectangle& operator=(const Rectangle& rect)
    {
      top_left = rect.top_left;
      top_right = rect.top_right;
      bottom_left = rect.bottom_left;
      bottom_right = rect.bottom_right;
      return *this;
    }

    void vertical_symetry()
    {
      Point2D tmp;
      tmp = top_left;
      top_left = top_right;
      top_right = tmp;

      tmp = bottom_left;
      bottom_left = bottom_right;
      bottom_right = tmp;
    }

    void horizontal_symetry()
    {
      Point2D tmp;
      tmp = top_left;
      top_left = bottom_left;
      bottom_left = tmp;

      tmp = top_right;
      top_right = bottom_right;
      bottom_right = tmp;
    }
  };
}

#endif /* !GEOMETRY_HH */