#ifndef GEOMETRY_HH
# define GEOMETRY_HH

namespace holovibes
{
  /*! Point in 2D */
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

    bool operator!=(const Point2D& p)
    {
      return x != p.x || y != p.y;
    }
  };

  /*! Rectangle used for selections */
  struct Rectangle
  {
    /* Even though 4 points do not guarantee that a rectangle
    is being represented, the ctor only takes two (top left and bot
    right). These 2 points always represent a rectangle. The two
    other points are built and the resulting structure will always
    be a rectangle*/
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

    unsigned int area()
    {
      unsigned int AB = top_right.x - top_left.x;
      unsigned int AD = top_left.y - bottom_left.y;
      return (AB * AD);
    }

    /*! \brief The two following functions will only be called in gui_gl_widget.cc
    ** in order to 'correctly' represent a selection rectangle.
    ** I.E. The first point clicked on during selection is considered as top
    ** left, but isn't necessarily the top left hand corner. These functions
    ** will readjust the rectangle so that the names are relevant.
    **/
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

    /*! \brief The two following functions will only be called in gui_gl_widget.cc
    ** in order to 'correctly' represent a selection rectangle.
    ** I.E. The first point clicked on during selection is considered as top
    ** left, but isn't necessarily the top left hand corner. These functions
    ** will readjust the rectangle so that the names are relevant.
    **/
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