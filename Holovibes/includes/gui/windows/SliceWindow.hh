/*! \file
 *
 * \brief Qt window containing the XZ or YZ view of the hologram.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
class MainWindow;

/*! \class SliceWindow
 *
 * \brief #TODO Add a description for this class
 */
class SliceWindow : public BasicOpenGLWindow
{
  public:
    SliceWindow(QPoint p,
                QSize s,
                DisplayQueue* q,
                KindOfView k,
                MainWindow* main_window = nullptr);
    virtual ~SliceWindow();

  protected:
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;
    MainWindow* main_window_;

    virtual void initShaders() override;
    virtual void initializeGL() override;
    virtual void paintGL() override;

    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void focusInEvent(QFocusEvent*) override;
};
} // namespace gui
} // namespace holovibes
