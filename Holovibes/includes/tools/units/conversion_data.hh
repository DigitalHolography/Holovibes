/*! \file
 *
 * \brief Implementation of the conversion from a unit to another
 */
#pragma once

#include "axis.hh"

namespace holovibes::gui
{
class BasicOpenGLWindow;
} // namespace holovibes::gui

namespace holovibes::units
{
/*! \class ConversionData
 *
 * \brief Encapsulates the conversion from a unit to another
 *
 * This will be copied a lot, make sure to keep references and pointers inside
 */
class ConversionData
{
  public:
    /*! \brief Constructs an object with the data needed to convert, to be modified for transforms */
    ConversionData(const gui::BasicOpenGLWindow& window);
    ConversionData(const gui::BasicOpenGLWindow* window);

    /*! \name Conversion between unit types
     * \{
     */
    float fd_to_opengl(int val, Axis axis) const;
    int opengl_to_fd(float val, Axis axis) const;
    /*! \} */

    void transform_from_fd(float& x, float& y) const;
    void transform_to_fd(float& x, float& y) const;

    const gui::BasicOpenGLWindow* get_opengl() const noexcept { return window_; }

  private:
    int get_fd_size(Axis axis) const;

    const gui::BasicOpenGLWindow* window_;
};
} // namespace holovibes::units
