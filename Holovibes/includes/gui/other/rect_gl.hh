/*! \file
 *
 * \brief Implementation of a Rectangle in OpenGL coordinate space [-1,1]
 */
#pragma once

#include <cmath>
#include <iostream>

#include "point_gl.hh"
#include "rect.hh"

namespace holovibes::gui
{
/*! \class Rect
 *
 * \brief A rectangle in OpenGL coordinate space [-1,1].
 *
 * It can be manipulated in two ways:
 * - top / bottom / left / right, making sure left < right and so on.
 * - x / y / width / height, with width > 0 and height > 0
 */
class RectGL
{
  public:
    /*! \brief Default constructors, will crash when trying to convert it */
    RectGL() = default;

    /*! \brief Constructs a rectangle from two points
     *
     * \param[in] src Top left point
     * \param[in] dst Bottom right point
     */
    RectGL(PointGL src, PointGL dst)
        : src_(src)
        , dst_(dst)
    {
    }

    /*! \brief Constructs a rectangle from a rectangle in the frame descriptor space
     * and project it onto the given window (keep scaling and traslation)
     *
     * \param[in] window The window where the rectangle will be projected
     * \param[in] rect The rectangle in the frame descriptor coordinate space [0, max(fd.height, fd.width)]
     */
    RectGL(const BasicOpenGLWindow& window, units::RectFd rect)
    {
        setSrc(PointGL(window, rect.src()));
        setDst(PointGL(window, rect.dst()));
    }

    /*! \name Getters and setters
     * \{
     */
    PointGL topLeft() const { return PointGL(x(), y()); }

    PointGL bottomRight() const { return PointGL(right(), bottom()); }

    PointGL topRight() const { return PointGL(right(), y()); }

    PointGL bottomLeft() const { return PointGL(x(), bottom()); }

    PointGL size() const
    {
        PointGL res;

        res.x() = dst_.x() > src_.x() ? dst_.x() : src_.x();
        res.x() -= dst_.x() > src_.x() ? src_.x() : dst_.x();

        res.y() = dst_.y() > src_.y() ? dst_.y() : src_.y();
        res.y() -= dst_.y() > src_.y() ? src_.y() : dst_.y();

        return res;
    }

    float x() const { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    float& x() { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    void setX(float newx)
    {
        x() = newx;
    }

    float y() const { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    float& y() { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    void setY(float newy)
    {
        y() = newy;
    }

    float width() const { return size().x(); }

    float unsigned_width() const
    {
        return std::abs(width());
    }

    void setWidth(float w)
    {
        right() = x() + w;
    }

    float height() const { return size().y(); }

    float unsigned_height() const
    {
        return std::abs(height());
    }

    void setHeight(float h)
    {
        bottom() = y() + h;
    }

    float& bottom() { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    float bottom() const { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    void setBottom(float y)
    {
        bottom() = y;
    }

    float& right() { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    float right() const { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    void setRight(float x)
    {
        right() = x;
    }

    void setTopLeft(PointGL p)
    {
        setX(p.x());
        setY(p.y());
    }

    void setBottomRight(PointGL p)
    {
        setRight(p.x());
        setBottom(p.y());
    }

    void setSrc(PointGL p) { src_ = p; }

    void setDst(PointGL p) { dst_ = p; }

    PointGL src() const { return src_; }

    PointGL dst() const { return dst_; }

    PointGL& srcRef() { return src_; }

    PointGL& dstRef() { return dst_; }
    /*! \} */

    /*! \brief area, abs(width * height) */
    float area() const { return std::abs(width() * height()); }

    /*! \brief Center of the rectangle */
    PointGL center() const
    {
        float x = this->x();
        x += right();
        x /= 2;
        float y = this->y();
        y += bottom();
        y /= 2;
        PointGL res(x, y);
        return res;
    }

  private:
    PointGL src_;
    PointGL dst_;
};

inline std::ostream& operator<<(std::ostream& o, const RectGL& r)
{
    return o << '[' << r.src() << ", " << r.dst() << ']';
}

inline bool operator==(const RectGL& lhs, const RectGL& rhs)
{
    return lhs.src() == rhs.src() && lhs.dst() == rhs.dst();
}
} // namespace holovibes::gui
