/*! \file
 *
 * \brief Implementation of a Rectangle
 */
#pragma once

#include "point.hh"
#include "window_pixel.hh"
#include "fd_pixel.hh"
#include "opengl_position.hh"
#include "json_macro.hh"

#include <cmath>

namespace holovibes::units
{
/*! \class Rect
 *
 * \brief A rectangle in some specific unit
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
    /*! \brief Default constructors, will crash when trying to convert it */
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

    /*! \brief Constructs a rectangle from its position and size */
    Rect(ConversionData data,
         typename T::primary_type x1 = 0,
         typename T::primary_type y1 = 0,
         typename T::primary_type x2 = 0,
         typename T::primary_type y2 = 0)
        : src_(data, x1, y1)
        , dst_(data, x2, y2)
    {
    }

    /*! \name Getters and setters
     * \{
     */
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
    /*! \} */

    /*! \brief Implicit cast into a rectangle of an other unit*/
    template <typename U>
    operator Rect<U>() const
    {
        Rect<U> res(src_, dst_);
        return res;
    }

    /*! \brief area, abs(width * height) */
    typename T::primary_type area() const { return std::abs(width() * height()); }

    /*! \brief Center of the rectangle */
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

  public:
    bool operator!=(const Rect& rhs) const { return false; }
};

/*! \brief Rectangle in the OpenGL coordinates [-1;1] */
using RectOpengl = Rect<OpenglPosition>;

/*! \brief Rectangle in the frame desc coordinates */
using RectFd = Rect<FDPixel>;

/*! \brief Rectangle in the window coordinates */
using RectWindow = Rect<WindowPixel>;

template <typename T>
std::ostream& operator<<(std::ostream& o, const Rect<T>& r)
{
    return o << '[' << r.src() << ", " << r.dst() << ']';
}

// clang-format off
SERIALIZE_JSON_ENUM(Axis, {
    {Axis::HORIZONTAL, "HORIZONTAL"},
    {Axis::VERTICAL, "VERTICAL"},
})

// Temporary situation needed to not touch all template classes in the units tools
inline void to_json(json& j, const units::RectFd& rect)
{
    j = json{
        {"src", {
            {"x", {
                {"val", rect.src().x().get()},
                {"axis", rect.src().x().get_axis()},
                {"conversion", (size_t)rect.src().x().getConversion().get_opengl()},
            }},
            {"y", {
                {"val", rect.src().y().get()},
                {"axis", rect.src().y().get_axis()},
                {"conversion", (size_t)rect.src().y().getConversion().get_opengl()},
            }},
        }},
        {"dst", {
            {"x", {
                {"val", rect.dst().x().get()},
                {"axis", rect.dst().x().get_axis()},
                {"conversion", (size_t)rect.dst().x().getConversion().get_opengl()},
            }},
            {"y", {
                {"val", rect.dst().y().get()},
                {"axis", rect.dst().y().get_axis()},
                {"conversion", (size_t)rect.dst().y().getConversion().get_opengl()},
            }},
        }}
    };
}

inline void from_json(const json& j, units::RectFd& rect)
{
    rect = units::RectFd(
        units::PointFd(
            units::FDPixel(
                (const gui::BasicOpenGLWindow*)j.at("src").at("x").at("conversion").get<size_t>(),
                j.at("src").at("x").at("axis").get<Axis>(),
                j.at("src").at("x").at("val").get<int>()
            ),
            units::FDPixel(
                (const gui::BasicOpenGLWindow*)j.at("src").at("y").at("conversion").get<size_t>(),
                j.at("src").at("y").at("axis").get<Axis>(),
                j.at("src").at("y").at("val").get<int>()
            )
        ),
        units::PointFd(
            units::FDPixel(
                (const gui::BasicOpenGLWindow*)j.at("dst").at("x").at("conversion").get<size_t>(),
                j.at("dst").at("x").at("axis").get<Axis>(),
                j.at("dst").at("x").at("val").get<int>()
            ),
            units::FDPixel(
                (const gui::BasicOpenGLWindow*)j.at("dst").at("y").at("conversion").get<size_t>(),
                j.at("dst").at("y").at("axis").get<Axis>(),
                j.at("dst").at("y").at("val").get<int>()
            )
        )
    );
}
// clang-format on
} // namespace holovibes::units