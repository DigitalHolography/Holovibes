/*! \file
 *
 * \brief declaration of the RainbowOverlay class.
 */
#pragma once

#include "Overlay.hh"
#include "unit.hh"

namespace holovibes::gui
{
/*! \class RainbowOverlay
 *
 * \brief class that represents a rainbow overlay in the window.
 */
class RainbowOverlay : public Overlay
{
  public:
    RainbowOverlay(BasicOpenGLWindow* parent);

    virtual ~RainbowOverlay() {}

    virtual void init() override;
    virtual void draw() override;

    void move(QMouseEvent* e) override;
    void release(ushort frameSide) override {}

  private:
    unsigned int check_interval(int x);

    void setBuffer() override;
};
} // namespace holovibes::gui
