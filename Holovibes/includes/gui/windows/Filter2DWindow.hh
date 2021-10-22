/*! \file
 *
 * \brief Qt window containing the Filter2D view of the hologram.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
class MainWindow;

/*! \class Filter2DWindow
 *
 * \brief #TODO Add a description for this class
 */
class Filter2DWindow : public BasicOpenGLWindow
{
  public:
    Filter2DWindow(QPoint p, QSize s, DisplayQueue* q);
    virtual ~Filter2DWindow();

  protected:
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;

    virtual void initShaders() override;
    virtual void initializeGL() override;
    virtual void paintGL() override;

    void focusInEvent(QFocusEvent*) override;
};
} // namespace gui
} // namespace holovibes
