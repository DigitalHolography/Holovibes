/*! \file
 *
 * \brief Implementation of a Unit with its cast between different coordinates system
 */
#pragma once

#include "axis.hh"
#include "frame_desc.hh"
#include "conversion_data.hh"

#include <iostream>

/*! \brief Contains functions and casts related to the three coordinates system. */
namespace holovibes::units
{

/*! \class Unit
 *
 * \brief A generic distance unit type
 *
 * Used to define implicit conversions between the different units
 * T will be either float or int, defined in the child classes
 */
template <typename T>
class Unit
{
  public:
    using primary_type = T;

    Unit(ConversionData data, Axis axis, T val)
        : conversion_data_(data)
        , axis_(axis)
        , val_(val)
    {
    }

    /*! \brief Implicit cast toward the primary type */
    operator T() const { return val_; }

    /*! \brief Implicit cast toward the primary type */
    operator T&() { return val_; }

    /*! \brief Implicit cast into a reference to the primary type
     *
     * Can be used for += and such
     */
    T& get() { return val_; }
    T get() const { return val_; }

    /*! \brief Exmplcit setter */
    void set(T x) { val_ = x; }

    /*! \brief Explicit setter */
    template <typename U>
    Unit<T> operator+(const U& other)
    {
        Unit<T> res(*this);
        res.val_ += other;
        return res;
    }

    const ConversionData& getConversion() const { return conversion_data_; }

    /*! \name Operator overloads
     * \{
     *
     * They can be used with either a primary type or another Unit
     * The result is an Unit, but can be implicitly casted into a T
     */
    template <typename U>
    Unit<T> operator-(const U& other)
    {
        Unit<T> res(*this);
        res.val_ -= other;
        return res;
    }

    template <typename U>
    Unit<T> operator/(const U& other)
    {
        Unit<T> res(*this);
        res.val_ /= other;
        return res;
    }

    template <typename U>
    Unit<T> operator*(const U& other)
    {
        Unit<T> res(*this);
        res.val_ *= other;
        return res;
    }

    template <typename U>
    Unit<T> operator-()
    {
        Unit<T> res(*this);
        res.val_ *= -1;
        return res;
    }
    /*! \} */

  protected:
    /*! \brief Encapsulates the metadata needed for the conversions */
    ConversionData conversion_data_;

    /*! \brief Which axis should be used when converting */
    Axis axis_;

    /*! \brief The value itself */
    T val_;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& o, const Unit<T>& x)
{
    return o << x.get();
}
} // namespace holovibes::units
