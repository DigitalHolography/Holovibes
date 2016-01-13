/*! \file 
 *
 * Store the 'Point' and the 'Rectangle' structures. */
#pragma once

namespace holovibes
{
  /*! Point in 2D */
  struct Point2D
  {
    int x;
    int y;

    /*! Construct a Point2D. */
    Point2D();
    /*! Construct a Point2D with another Point2D. */
    Point2D(const Point2D& p);
    /*! Construct a Point2D with two coordinates. */
    Point2D(const int xcoord, const int ycoord);
    /*! Assignement operator. */
    Point2D& operator=(const Point2D& p);
    /*! Unequality operator. */
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

    /*! Construct a Rectangle. */
    Rectangle();
    /*! Construct a Rectangle with two points. */
    Rectangle(const Point2D& top_left_corner, const Point2D& bottom_right_corner);
    /*! Construct a Rectangle with another Rectangle. */
    Rectangle(const Rectangle& rect);

    /*! Assignement operator. */
    Rectangle& operator=(const Rectangle& rect);

    /*! Compute the area of the Rectangle. */
    unsigned int area() const;
    /*! Return the width of the Rectangle. */
    unsigned int get_width() const;
    /*! Return the height of the Rectangle. */
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