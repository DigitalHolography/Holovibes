/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Interface for all rectangular overlays. */
#pragma once

#include "Overlay.hh"

namespace holovibes
{
namespace gui
{
class RectOverlay : public Overlay
{
  public:
    RectOverlay(KindOfOverlay overlay, BasicOpenGLWindow* parent);

    virtual ~RectOverlay() {}

    virtual void init() override;
    virtual void draw() override;

    virtual void move(QMouseEvent* e) override;
    virtual void checkCorners();

  protected:
    void setBuffer() override;
};
} // namespace gui
} // namespace holovibes
