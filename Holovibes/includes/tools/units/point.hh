/*! \file
 *
 * \brief Implementation of a Point
 */
#pragma once

#include "conversion_data.hh"
#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"
#include "real_position.hh"

#include <type_traits>

namespace holovibes
{
namespace units
{

/*! \brief A point in some specific unit
 */
template <class T>
class Point
{
  public:
    Point()
        : x_(ConversionData(nullptr), Axis::HORIZONTAL, 0)
        , y_(ConversionData(nullptr), Axis::VERTICAL, 0)
    {
    }

    Point(T x, T y)
        : x_(x)
        , y_(y)
    {
    }

    /*! \brief Constructs a point from the needed conversion data and two
     * primary types
     */
    Point(ConversionData data,
          typename T::primary_type x = 0,
          typename T::primary_type y = 0)
        : x_(data, Axis::HORIZONTAL, x)
        , y_(data, Axis::VERTICAL, y)
    {
    }

    T& x() { return x_; }

    T& y() { return y_; }

    T x() const { return x_; }

    T y() const { return y_; }

    /*! \brief Implicit cast into a point of an other unit
     */
    template <typename U>
    operator Point<U>() const
    {
        Point<OpenglPosition> tmp(x_, y_);
        // We can't use "if constexpr" here because cuda isn't c++17
        // Once it is, please add the constexpr
        if (std::is_same<T, FDPixel>::value)
            x_.getConversion().transform_from_fd(tmp.x(), tmp.y());
        if (std::is_same<U, FDPixel>::value)
            x_.getConversion().transform_to_fd(tmp.x(), tmp.y());
        Point<U> res(tmp.x(), tmp.y());
        return res;
    }

    // Casting to RealPosition only when we are not already in RealPosition
    template <
        typename = std::enable_if_t<(!(std::is_same<RealPosition, T>::value))>>
    operator Point<RealPosition>() const
    {
        Point<RealPosition> res(x_, y_);
        return res;
    }
    /*! \brief Operator overloads
     */
    /**@{*/
    Point<T> operator+(const Point<T>& other) const
    {
        Point<T> res(x_, y_);
        res.x_ += other.x_;
        res.y_ += other.y_;
        return res;
    }

    Point<T> operator-(const Point<T>& other) const
    {
        Point<T> res(x_, y_);
        res.x_ -= other.x_;
        res.y_ -= other.y_;
        return res;
    }

    double distance() const { return sqrt(pow(x_, 2) + pow(y_, 2)); }

    bool operator==(const Point<T>& other) const
    {
        return x_ == other.x_ && y_ == other.y_;
    }
    /**@}*/

  private:
    T x_;
    T y_;
};

/*! \brief A point in the OpenGL coordinate system [-1;1]
 */
using PointOpengl = Point<OpenglPosition>;

/*! \brief A point in the frame desc coordinate system [0;fd.width]
 */
using PointFd = Point<FDPixel>;

/*! \brief A point in the window coordinate system [0;window size]
 */
using PointWindow = Point<WindowPixel>;

using PointReal = Point<RealPosition>;

template <typename T>
std::ostream& operator<<(std::ostream& o, const Point<T>& p)
{
    return o << '(' << p.x() << ", " << p.y() << ')';
}

} // namespace units
} // namespace holovibes
