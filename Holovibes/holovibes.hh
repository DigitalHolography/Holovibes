#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera.hh"
# include "thread_gl_window.hh"
# include "thread_capture.hh"

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

  private:
    camera::Camera* camera_;
    ThreadCapture* tcapture_;
    ThreadGLWindow* tglwnd_;
  };
}

#endif /* !HOLOVIBES_HH */
