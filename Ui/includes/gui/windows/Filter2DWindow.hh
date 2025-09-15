/*! \file
 *
 * \brief Qt window containing the Filter2D view of the hologram.
 */
#pragma once

#include "BasicOpenGLWindow.hh"
#include "CudaTexture.hh"
#include "TextureWindowHelper.hh"
namespace holovibes::gui
{
class MainWindow;

/*! \class Filter2DWindow
 *
 * \brief Class that represents a Filter2D window in the GUI.
 */
class Filter2DWindow : public TextureWindowHelper
{
  public:
    Filter2DWindow(QPoint p, QSize s, DisplayQueue* q);
    virtual ~Filter2DWindow();

  protected:
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;



    void focusInEvent(QFocusEvent*) override;
    void closeEvent(QCloseEvent*) override;
};
} // namespace holovibes::gui
