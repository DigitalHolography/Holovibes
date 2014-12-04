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
    void set_average_mode(bool value);

  protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;

  private:
    void selection_rect(const QRect& selection, float color[4]);
    void zoom(const QRect& selection);
    void dezoom();

    /* Assure that the rectangle starts at topLeft and ends at bottomRight
    no matter what direction the user uses to select a zone */
    void swap_selection_corners(QRect& selection);

    void gl_error_checking();

    // Debug
    void print_selection(QRect& selection);

  private:
    QTimer timer_;
    bool is_selection_enabled_;
    QRect selection_;
    bool is_zoom_enabled_;
    bool is_average_enabled_;
    bool is_signal_selection_;
    QRect signal_selection_;
    QRect noise_selection_;

    // Translation
    float px_;
    float py_;

    // Zoom ratio
    float zoom_ratio_;

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