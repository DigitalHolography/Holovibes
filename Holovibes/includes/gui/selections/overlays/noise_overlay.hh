/*! \file
 *
 * \brief Overlay selecting the noise zone to chart.
 */
#pragma once

#include "filled_rect_overlay.hh"

namespace holovibes::gui
{
/*! \class NoiseOverlay
 *
 * \brief #TODO Add a description for this class
 */
class NoiseOverlay : public FilledRectOverlay
{
  public:
    NoiseOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
