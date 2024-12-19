/*! \file
 *
 * \brief Qt window containing the XZ or YZ view of the hologram.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{
class MainWindow;

/*! \class SliceWindow
 *
 * \brief class that represents a slice window in the GUI.
 */
class SliceWindow : public BasicOpenGLWindow
{
  public:
    SliceWindow(QPoint p, QSize s, DisplayQueue* q, KindOfView k);
    virtual ~SliceWindow();

  protected:
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;

    void initShaders() override;
    void initializeGL() override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void focusInEvent(QFocusEvent*) override;
    void closeEvent(QCloseEvent*) override;
};
} // namespace holovibes::gui
