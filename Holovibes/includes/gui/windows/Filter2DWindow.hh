/*! \file
 *
 * \brief Qt window containing the Filter2D view of the hologram.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
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

    void initShaders() override;
    void initializeGL() override;
    void paintGL() override;

    void focusInEvent(QFocusEvent*) override;
    void closeEvent(QCloseEvent*) override;
};
} // namespace holovibes::gui
