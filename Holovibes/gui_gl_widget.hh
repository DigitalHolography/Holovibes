#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <QGLWidget>
# include <GL/GL.h>
# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "camera.hh"
# include "queue.hh"

namespace gui
{
  class GLWidget : public QGLWidget
  {
    Q_OBJECT

  public:
    GLWidget(QWidget *parent, holovibes::Queue& q);
    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

  protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

  private:
    holovibes::Queue& q_;
    camera::FrameDescriptor fd_;
    GLuint texture_;
    void* frame_;
  };
}

#endif /* !GUI_GL_WIDGET_HH_ */