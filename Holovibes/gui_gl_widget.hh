#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <QGLWidget>
# include <QOpenGLFunctions.h>
# include <QTimer>
# include <QMouseEvent>

# include <cuda_gl_interop.h>

# include "queue.hh"
# include "frame_desc.hh"

namespace gui
{
  class GLWidget : public QGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT
      const unsigned int DISPLAY_FRAMERATE = 30;

  public:
    GLWidget(
      holovibes::Queue& q,
      unsigned int width,
      unsigned int height,
      QWidget* parent = 0);
    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

  public slots:
    void resizeFromWindow(int width, int height);

  protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

  private:
    void selection_rect(int startx, int starty, int endx, int endy, float color[4]);
    void gl_error_checking();

  private:
    /* --- QT --- */
    QTimer timer_;
    bool is_selection_enabled_;
    unsigned int startx_;
    unsigned int starty_;
    unsigned int endx_;
    unsigned int endy_;

    /* Window size hints */
    unsigned int width_;
    unsigned int height_;

    /* --- CUDA/OpenGL --- */
    holovibes::Queue& queue_;
    const camera::FrameDescriptor& frame_desc_;

    GLuint buffer_;
    struct cudaGraphicsResource* cuda_buffer_;
  };
}

#endif /* !GUI_GL_WIDGET_HH_ */