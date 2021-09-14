/*! \file
 *
 * \brief Qt window used to display the input frames.
 */
#pragma once

#include "BasicOpenGLWindow.hh"

namespace holovibes
{
namespace gui
{
class SliceWindow;

/*! \class RawWindow
 *
 * \brief #TODO Add a description for this class
 */
class RawWindow : public BasicOpenGLWindow
{
  public:
    RawWindow(QPoint p,
              QSize s,
              DisplayQueue* q,
              KindOfView k = KindOfView::Raw);
    virtual ~RawWindow();

    void zoomInRect(units::RectOpengl zone);
    void setRatio(float ratio_);

    bool is_resize_call() const;
    void set_is_resize(bool b);

  protected:
    int texDepth, texType;

    int old_width = -1;
    int old_height = -1;
    // it represents width/height of the Raw window
    float ratio = 0.0f;

    // bool represent if we are resizing the window or creating one
    bool is_resize = true;

    const float translation_step_ = 0.05f;

    virtual void initShaders() override;
    virtual void initializeGL() override;
    virtual void resizeGL(int width, int height) override;
    virtual void paintGL() override;

    void mousePressEvent(QMouseEvent* e);
    void mouseMoveEvent(QMouseEvent* e);
    void mouseReleaseEvent(QMouseEvent* e);
    void keyPressEvent(QKeyEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
};
} // namespace gui
} // namespace holovibes
