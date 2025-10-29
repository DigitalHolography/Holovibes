/*! \file
 *
 * \brief Overlay used to edit the off-axis Filter2D mask.
 */
#pragma once

#include "square_overlay.hh"

namespace holovibes::gui
{
class Filter2DOffAxisOverlay : public SquareOverlay
{
  public:
    explicit Filter2DOffAxisOverlay(BasicOpenGLWindow* parent);

    void onSetCurrent() override;
    void release(ushort frameSide) override;

    /*! \brief Update overlay coordinates from current settings. */
    void apply_settings();

  private:
    void ensure_square_within_bounds();
};
} // namespace holovibes::gui
