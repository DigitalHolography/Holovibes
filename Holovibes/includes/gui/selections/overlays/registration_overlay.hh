/*! \file
 *
 * \brief Overlay used to display a circle for the registration mask at the center of the window.
 */
#pragma once

#include "circ_overlay.hh"

namespace holovibes::gui
{
/*! \class RegistrationOverlay
 *
 * \brief Overlay used to display a circle for the registration mask at the center of the window.
 */
class RegistrationOverlay : public CircOverlay
{
  public:
    RegistrationOverlay(BasicOpenGLWindow* parent);
    virtual ~RegistrationOverlay() {}

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;
};
} // namespace holovibes::gui
