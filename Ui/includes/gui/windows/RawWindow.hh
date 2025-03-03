/*! \file
 *
 * \brief Qt window used to display the input frames.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

#include "rect.hh"

namespace holovibes::gui
{

class SliceWindow;

/*! \class RawWindow
 *
 * \brief Class that represents a raw window in the GUI.
 */
class RawWindow : public BasicOpenGLWindow
{
  public:
    RawWindow(QPoint p, QSize s, DisplayQueue* q, float ratio = 0.f, KindOfView k = KindOfView::Raw);
    virtual ~RawWindow();

    void zoomInRect(units::RectFd zone);

    bool is_resize_call() const;
    void set_is_resize(bool b);

    void save_gui(std::string window);

  protected:
    int texDepth, texType;
    cudaArray_t cuArray;
    cudaResourceDesc cuArrRD;
    cudaSurfaceObject_t cuSurface;

    int old_width = -1;
    int old_height = -1;
    /*! \brief Width/height ratio of the Raw window */
    float ratio = 0.0f;

    /*! If we are resizing the window or creating one */
    bool is_resize = true;

    const float translation_step_ = 0.05f;

    void initShaders() override;
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void keyPressEvent(QKeyEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;

    void closeEvent(QCloseEvent* event) override;
};
} // namespace holovibes::gui
