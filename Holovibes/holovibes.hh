#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera.hh"
# include "thread_gl_window.hh"
# include "thread_capture.hh"
# include "recorder.hh"

namespace holovibes
{
  class Holovibes
  {
  public:
    enum camera_type
    {
      PIKE,
      XIQ,
      IDS,
      PIXELFLY,
    };

    Holovibes(enum camera_type c);
    ~Holovibes();

    void init_display(
      unsigned int w,
      unsigned int h);
    void dispose_display();

    void init_capture(unsigned int buffer_nb_elts);
    void dispose_capture();

    void init_recorder(
      std::string& filepath,
      unsigned int rec_n_images);
    void dispose_recorder();

  private:
    camera::Camera* camera_;
    ThreadCapture* tcapture_;
    ThreadGLWindow* tglwnd_;
    Recorder* recorder_;
  };
}

#endif /* !HOLOVIBES_HH */
