/*! \file
 *
 * \brief Implementation of a Rectangle
 */
#pragma once

#include <cmath>

#include "all_struct.hh"
#include "point.hh"

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
class RectFd
{
  public:
    /*! \brief Constructs a rectangle from two points
     *
     * \param top_left Top left point
     * \param size bottom_right Bottom right point
     */
    RectFd(PointFd src, PointFd dst)
        : src_(src)
        , dst_(dst)
    {
    }

    /*! \brief Constructs a rectangle from its position and size */
    RectFd(int x1 = 0, int y1 = 0, int x2 = 0, int y2 = 0)
        : src_(x1, y1)
        , dst_(x2, y2)
    {
    }

    /*! \name Getters and setters
     * \{
     */
    PointFd topLeft() const { return PointFd(x(), y()); }

    PointFd bottomRight() const { return PointFd(right(), bottom()); }

    PointFd topRight() const { return PointFd(right(), y()); }

    PointFd bottomLeft() const { return PointFd(x(), bottom()); }

    PointFd size() const
    {
        PointFd res;

        res.x() = dst_.x() > src_.x() ? dst_.x() : src_.x();
        res.x() -= dst_.x() > src_.x() ? src_.x() : dst_.x();

        res.y() = dst_.y() > src_.y() ? dst_.y() : src_.y();
        res.y() -= dst_.y() > src_.y() ? src_.y() : dst_.y();

        return res;
    }

    int x() const { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    int& x() { return src_.x() < dst_.x() ? src_.x() : dst_.x(); }

    void setX(int newx) { x() = newx; }

    int y() const { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    int& y() { return src_.y() < dst_.y() ? src_.y() : dst_.y(); }

    void setY(int newy) { y() = newy; }

    int width() const { return size().x(); }

    int unsigned_width() const { return std::abs(width()); }

    void setWidth(int w) { right() = x() + w; }

    int height() const { return size().y(); }

    int unsigned_height() const { return std::abs(height()); }

    void setHeight(int h) { bottom() = y() + h; }

    int& bottom() { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    int bottom() const { return src_.y() > dst_.y() ? src_.y() : dst_.y(); }

    void setBottom(int y) { bottom() = y; }

    int& right() { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    int right() const { return src_.x() > dst_.x() ? src_.x() : dst_.x(); }

    void setRight(int x) { right() = x; }

    void setTopLeft(PointFd p)
    {
        setX(p.x());
        setY(p.y());
    }

    void setBottomRight(PointFd p)
    {
        setRight(p.x());
        setBottom(p.y());
    }

    void setSrc(PointFd p) { src_ = p; }

    void setDst(PointFd p) { dst_ = p; }

    PointFd src() const { return src_; }

    PointFd dst() const { return dst_; }

    PointFd& srcRef() { return src_; }

    PointFd& dstRef() { return dst_; }
    /*! \} */

    /*! \brief area, abs(width * height) */
    int area() const { return std::abs(width() * height()); }

    /*! \brief Center of the rectangle */
    PointFd center() const
    {
        int x = this->x();
        x += right();
        x /= 2;
        int y = this->y();
        y += bottom();
        y /= 2;
        PointFd res(x, y);
        return res;
    }

    inline bool operator==(const units::RectFd& other) const { return src_ == other.src_ && dst_ == other.dst_; }

    SERIALIZE_JSON_STRUCT(RectFd, src_, dst_);

  private:
    PointFd src_;
    PointFd dst_;
};

inline std::ostream& operator<<(std::ostream& o, const RectFd& r)
{
    return o << '[' << r.src() << ", " << r.dst() << ']';
}
} // namespace holovibes::units
