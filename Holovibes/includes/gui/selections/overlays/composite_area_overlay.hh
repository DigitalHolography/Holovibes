/*! \file
 *
 * \brief Overlay selecting the zone to normalize colors.
 */
#pragma once

#include "square_overlay.hh"

namespace holovibes
{
namespace gui
{
/*! \class CompositeAreaOverlay
 *
 * \brief #TODO Add a description for this class
 */
class CompositeAreaOverlay : public RectOverlay
{
  public:
    CompositeAreaOverlay(BasicOpenGLWindow* parent);

    void release(ushort frameSide) override;
};
} // namespace gui
} // namespace holovibes
