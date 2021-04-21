/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Qt window containing the Filter2D view of the hologram. */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
class MainWindow;

class Filter2DWindow : public BasicOpenGLWindow
{
  public:
    Filter2DWindow(QPoint p,
                QSize s,
                DisplayQueue* q,
                MainWindow* main_window = nullptr);
    virtual ~Filter2DWindow();

  protected:
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;
    MainWindow* main_window_;

    virtual void initShaders() override;
    virtual void initializeGL() override;
    virtual void paintGL() override;

    void focusInEvent(QFocusEvent*) override;
};
} // namespace gui
} // namespace holovibes
