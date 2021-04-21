/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Overlay used to display a reticle in the center of the window. */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "Overlay.hh"

namespace holovibes
{
namespace gui
{
class Filter2DReticleOverlay : public Overlay
{
  public:
    Filter2DReticleOverlay(BasicOpenGLWindow* parent);
    virtual ~Filter2DReticleOverlay() {}

    void init() override;
    void draw() override;

    virtual void move(QMouseEvent* e) override {}
    virtual void release(ushort frameside) override {}

  protected:
    void setBuffer() override;

    //! Transparency of the lines
    float alpha_;
};
} // namespace gui
} // namespace holovibes
