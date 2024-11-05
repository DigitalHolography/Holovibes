/*! \file
 *
 * \brief Interface for all filled rectangular overlays.
 *
 * You can control:
 * - The color of the edges with the variable `color_`.
 * - The opacity of the edges with the variable `alpha_`.
 * - The opacity of the fill color with the variable `fill_alpha_`.
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class FilledRectOverlay
 *
 * \brief A class that implement filled rectangular overlays.
 */
class FilledRectOverlay : public RectOverlay
{
  public:
    FilledRectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);
    virtual ~FilledRectOverlay();

    virtual void init() override;
    virtual void draw() override;

  protected:
    /*! \brief Transparency of the overlay, between 0 and 1 */
    float fill_alpha_;

    /*! \brief Indexes buffers for opengl used to tell in which order to render the inside of the rect */
    GLuint fill_elem_index_;
};
} // namespace holovibes::gui
