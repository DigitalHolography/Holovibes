#pragma once

namespace holovibes
{
  /*! Point in 2D */
  struct Point2D
  {
    int x;
    int y;

    Point2D();
    Point2D(const Point2D& p);
    Point2D(const int xcoord, const int ycoord);

    Point2D& operator=(const Point2D& p);
    bool operator!=(const Point2D& p);
  };

  /*! \brief Rectangle used for selections
   *
   * Even though 4 points do not guarantee that a rectangle
   * is being represented, the vector only takes two (top left and bot
   * right). These 2 points always represent a rectangle. The two
   * other points are built and the resulting structure will always
   * be a rectangle
   */
  struct Rectangle
  {
    Point2D top_left;
    Point2D top_right;
    Point2D bottom_left;
    Point2D bottom_right;

    Rectangle();
    Rectangle(const Point2D& top_left_corner, const Point2D& bottom_right_corner);
    Rectangle(const Rectangle& rect);

    Rectangle& operator=(const Rectangle& rect);

    unsigned int area() const;
    unsigned int get_width() const;
    unsigned int get_height() const;

    /*! \brief The two following functions will only be called in gui_gl_widget.cc
    **
    ** in order to 'correctly' represent a selection rectangle.
    ** I.E. The first point clicked on during selection is considered as top
    ** left, but isn't necessarily the top left hand corner. These functions
    ** will readjust the rectangle so that the names are relevant.
    **/
    void vertical_symetry();

    /*! \brief The two following functions will only be called in gui_gl_widget.cc
    **
    ** in order to 'correctly' represent a selection rectangle.
    ** I.E. The first point clicked on during selection is considered as top
    ** left, but isn't necessarily the top left hand corner. These functions
    ** will readjust the rectangle so that the names are relevant.
    **/
    void horizontal_symetry();
  };
}