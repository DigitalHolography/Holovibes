/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Implementation of a Rectangle */
#pragma once

#include "point.hh"
#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"

#include <cmath>

namespace holovibes
{
namespace units
{

/*! \brief A rectangle in some specific unit
 *
 * It can be manipulated in two ways:
 * through top / bottom / left / right, making sure left < right and so on
 * or as source / destination, two corner points that can be swapped (used in
 * overlays)
 */
template <typename T>
class Rect
{
  public:
    /*! \brief Default constructors, will crash when trying to convert it
     */
    Rect() = default;

    /*! \brief Constructs a rectangle from two points
     *
     * \param top_left Top left point
     * \param size bottom_right Bottom right point
     */
    Rect(Point<T> src, Point<T> dst)
        : src_(src)
        , dst_(dst)
    {
    }

    /*! \brief Constructs a rectangle from its position and size
     */
    Rect(ConversionData data,
         typename T::primary_type x1 = 0,
         typename T::primary_type y1 = 0,
         typename T::primary_type x2 = 0,
         typename T::primary_type y2 = 0)
        : src_(data, x1, y1)
        , dst_(data, x2, y2)
    {
    }

    /*! \brief Getters and setters
     */
    /**@{*/
    Point<T> topLeft() const { return Point<T>(x(), y()); }

    Point<T> bottomRight() const { return Point<T>(right(), bottom()); }

    Point<T> topRight() const { return Point<T>(right(), y()); }

    Point<T> bottomLeft() const { return Point<T>(x(), bottom()); }

    Point<T> size() const
    {
        Point<T> res;

        res.x() = dst_.x() > src_.x() ? dst_.x() : src_.x();
        res.x() -= dst_.x() > src_.x() ? src_.x() : dst_.x();

        res.y() = dst_.y() > src_.y() ? dst_.y() : src_.y();
        res.y() -= dst_.y() > src_.y() ? src_.y() : dst_.y();

        return res;
    }

    T x() const { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    T& x() { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    template <typename U>
    void setX(U newx)
    {
        x().set(newx);
    }

    T y() const { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    T& y() { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    template <typename U>
    void setY(U newy)
    {
        y().set(newy);
    }

    T width() const { return size().x(); }

    T unsigned_width() const
    {
        T res = size().x();
        if (res < 0)
            res *= -1;
        return res;
    }

    template <typename U>
    void setWidth(U w)
    {
        right().set(x() + w);
    }

    T height() const { return size().y(); }

    T unsigned_height() const
    {
        T res = size().y();
        if (res < 0)
            res *= -1;
        return res;
    }

    template <typename U>
    void setHeight(U h)
    {
        bottom().set(y() + h);
    }

    T& bottom() { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    T bottom() const { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    template <typename U>
    void setBottom(U y)
    {
        bottom().set(y);
    }

    T& right() { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    T right() const { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    template <typename U>
    void setRight(U x)
    {
        return right().set(x);
    }

    void setTopLeft(Point<T> p)
    {
        setX(p.x());
        setY(p.y());
    }

    void setBottomRight(Point<T> p)
    {
        setRight(p.x());
        setBottom(p.y());
    }

    void setSrc(Point<T> p) { src_ = p; }

    void setDst(Point<T> p) { dst_ = p; }

    Point<T> src() const { return src_; }

    Point<T> dst() const { return dst_; }

    Point<T>& srcRef() { return src_; }

    Point<T>& dstRef() { return dst_; }

    /**@}*/

    /*! \brief Implicit cast into a rectangle of an other unit
     */
    template <typename U>
    operator Rect<U>() const
    {
        Rect<U> res(src_, dst_);
        return res;
    }

    /*! \brief area, abs(width * height)
     */
    typename T::primary_type area() const
    {
        return std::abs(width() * height());
    }

    /*! \brief Center of the rectangle
     */
    Point<T> center() const
    {
        T x = this->x();
        x += right();
        x /= 2;
        T y = this->y();
        y += bottom();
        y /= 2;
        Point<T> res(x, y);
        return res;
    }

  private:
    Point<T> src_;
    Point<T> dst_;
};

/*! \brief Rectangle in the OpenGL coordinates [-1;1]
 */
using RectOpengl = Rect<OpenglPosition>;

/*! \brief Rectangle in the frame desc coordinates
 */
using RectFd = Rect<FDPixel>;

/*! \brief Rectangle in the window coordinates
 */
using RectWindow = Rect<WindowPixel>;

template <typename T>
std::ostream& operator<<(std::ostream& o, const Rect<T>& r)
{
    return o << '[' << r.src() << ", " << r.dst() << ']';
}

} // namespace units
} // namespace holovibes
