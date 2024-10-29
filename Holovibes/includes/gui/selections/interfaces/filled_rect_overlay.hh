/*! \file
 *
 * \brief Interface for all rectangular overlays that are filled (can specify the opacity of the edges and the fill).
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class FilledRectOverlay
 *
 * \brief A class that implement rectangular overlays that are filled (can specify the opacity of the edges and the
 * fill).
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
