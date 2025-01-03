/*! \file
 *
 * \brief Implementation of a Point
 */
#pragma once

#include <iostream>

#include "all_struct.hh"

namespace holovibes::units
{
/*! \class Point
 *
 * \brief A point in some specific unit
 */
class PointFd
{
  public:
    PointFd()
        : x_(0)
        , y_(0)
    {
    }

    PointFd(int x, int y)
        : x_(x)
        , y_(y)
    {
    }

    int& x() { return x_; }

    int& y() { return y_; }

    int x() const { return x_; }

    int y() const { return y_; }

    /*! \name Operator overloads
     * \{
     */
    PointFd operator+(const PointFd& other) const
    {
        PointFd res(x_, y_);
        res.x_ += other.x_;
        res.y_ += other.y_;
        return res;
    }

    PointFd operator-(const PointFd& other) const
    {
        PointFd res(x_, y_);
        res.x_ -= other.x_;
        res.y_ -= other.y_;
        return res;
    }

    inline bool operator==(const PointFd& other) const { return x_ == other.x_ && y_ == other.y_; }
    /*! \} */

    SERIALIZE_JSON_STRUCT(PointFd, x_, y_);

  private:
    int x_;
    int y_;
};

inline std::ostream& operator<<(std::ostream& o, const PointFd& p)
{
    return o << '(' << p.x() << ", " << p.y() << ')';
}
} // namespace holovibes::units
