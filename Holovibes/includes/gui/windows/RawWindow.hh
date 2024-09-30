/*! \file
 *
 * \brief Qt window used to display the input frames.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes::gui
{

class SliceWindow;

/*! \class RawWindow
 *
 * \brief #TODO Add a description for this class
 */
class RawWindow : public BasicOpenGLWindow
{
  public:
    RawWindow(QPoint p, QSize s, DisplayQueue* q, float ratio = 0.f, KindOfView k = KindOfView::Raw);
    virtual ~RawWindow();

    void zoomInRect(units::RectOpengl zone);

    bool is_resize_call() const;
    void set_is_resize(bool b);

    void save_gui(std::string window);

  protected:
    int texDepth, texType;

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
