/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "Overlay.hh"
#include "unit.hh"

namespace holovibes
{
namespace gui
{
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
    int check_interval(int x);

    void setBuffer() override;
};
} // namespace gui
} // namespace holovibes