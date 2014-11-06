#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <QGLWidget>
# include "camera.hh"
# include <GL/GL.h>

namespace gui
{
  class GLWidget : public QGLWidget
  {
    Q_OBJECT

  public:
    GLWidget(QWidget *parent, camera::FrameDescriptor fd);
    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    // debug
    void setFrame(void* frame);

  protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

  private:
    camera::FrameDescriptor fd_;
    GLuint texture_;
    void* frame_;
  };
}

#endif /* !GUI_GL_WIDGET_HH_ */