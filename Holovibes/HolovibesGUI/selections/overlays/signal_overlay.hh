/*! \file
 *
 * \brief Overlay selecting the signal zone to chart.
 */
#pragma once

#include "rect_overlay.hh"

namespace holovibes::gui
{
/*! \class SignalOverlay
 *
 * \brief #TODO Add a description for this class
 */
class SignalOverlay : public RectOverlay
{
  public:
    SignalOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
