#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <QGLWidget>
# include <QTimer>
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
    GLWidget(holovibes::Queue& q, unsigned int width, unsigned int height, QWidget* parent = 0);
    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

  public slots:
    void resizeFromWindow(int width, int height);

  protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

  private:
    holovibes::Queue& q_;
    unsigned int width_;
    unsigned int height_;
    camera::FrameDescriptor fd_;
    GLuint texture_;
    void* frame_;
    QTimer timer_;
  };
}

#endif /* !GUI_GL_WIDGET_HH_ */