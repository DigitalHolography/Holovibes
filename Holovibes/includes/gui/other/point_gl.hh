/*! \file
 *
 * \brief Contains a 2D point in the opengl system.
 */
#pragma once

#include <iostream>

#include "point.hh"
#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
/*! \class PointGL
 *
 * \brief A single 2D point in the OpenGL coordinate system [-1, 1].
 */
class PointGL
{
  public:
    /*! \brief Defaults ctor sets to 0 */
    PointGL()
        : x_(0)
        , y_(0)
    {
    }

    /*! \brief Constructs a point from two float
     *
     * \param[in] x The x coordinate in range [-1, 1]
     * \param[in] y The y coordinate in range [-1, 1]
     */
    PointGL(float x, float y)
        : x_(x)
        , y_(y)
    {
    }

    /*! \brief Constructs a point from a point in the frame descriptor space
     * and project it onto the given window (keep scaling and traslation)
     *
     * \param[in] window The window where the point will be projected
     * \param[in] point The point in the frame descriptor coordinate space [0, max(fd.height, fd.width)]
     */
    PointGL(const BasicOpenGLWindow& window, units::PointFd point);

    float& x() { return x_; }

    float& y() { return y_; }

    float x() const { return x_; }

    float y() const { return y_; }

    /*! \name Operator overloads
     * \{
     */
    PointGL operator+(const PointGL& other) const
    {
        PointGL res(x_, y_);
        res.x_ += other.x_;
        res.y_ += other.y_;
        return res;
    }

    PointGL operator-(const PointGL& other) const
    {
        PointGL res(x_, y_);
        res.x_ -= other.x_;
        res.y_ -= other.y_;
        return res;
    }

    units::PointFd to_fd() const;

    bool operator==(const PointGL& other) const { return x_ == other.x_ && y_ == other.y_; }

private:
  float x_;
  float y_;
};

inline std::ostream& operator<<(std::ostream& o, const PointGL& p)
{
    return o << '(' << p.x() << ", " << p.y() << ')';
}
} // namespace holovibes::gui
