#ifndef GUI_GL_WIDGET_HH_
# define GUI_GL_WIDGET_HH_

# include <QGLWidget>
# include <QOpenGLFunctions.h>
# include <QTimer>
# include <QMouseEvent>

# include <cuda_gl_interop.h>

# include "queue.hh"
# include <frame_desc.hh>
# include "geometry.hh"
# include "holovibes.hh"

namespace gui
{
  class GLWidget : public QGLWidget, protected QOpenGLFunctions
  {
    Q_OBJECT
      const unsigned int DISPLAY_FRAMERATE = 30;

  public:
    GLWidget(
      holovibes::Holovibes& h,
      holovibes::Queue& q,
      unsigned int width,
      unsigned int height,
      QWidget* parent = 0);
    ~GLWidget();
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    void enable_selection()
    {
      is_selection_enabled_ = true;
    }

    const holovibes::Rectangle& get_signal_selection() const
    {
      return signal_selection_;
    }

    const holovibes::Rectangle& get_noise_selection() const
    {
      return noise_selection_;
    }

    void set_signal_selection(const holovibes::Rectangle& selection)
    {
      signal_selection_ = selection;
      h_.get_compute_desc().signal_zone = signal_selection_;
    }

    void set_noise_selection(const holovibes::Rectangle& selection)
    {
      noise_selection_ = selection;
      h_.get_compute_desc().noise_zone = noise_selection_;
    }

    void launch_average_computation();

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
    void selection_rect(const holovibes::Rectangle& selection, float color[4]);
    void zoom(const holovibes::Rectangle& selection);
    void dezoom();

    /* Assure that the rectangle starts at topLeft and ends at bottomRight
    no matter what direction the user uses to select a zone */
    void swap_selection_corners(holovibes::Rectangle& selection);

    void gl_error_checking();

  private:
    holovibes::Holovibes& h_;

    QTimer timer_;
    bool is_selection_enabled_;
    holovibes::Rectangle selection_;
    bool is_zoom_enabled_;
    bool is_average_enabled_;
    bool is_signal_selection_;
    holovibes::Rectangle signal_selection_;
    holovibes::Rectangle noise_selection_;
    QWidget* parent_;

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