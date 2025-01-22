/*! \file
 *
 * \brief Overlay selecting the signal zone to chart.
 */
#pragma once

#include "filled_rect_overlay.hh"

namespace holovibes::gui
{
/*! \class SignalOverlay
 *
 * \brief Class that represents a signal overlay in the window.
 */
class SignalOverlay : public FilledRectOverlay
{
  public:
    SignalOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace holovibes::gui
