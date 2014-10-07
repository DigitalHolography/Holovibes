#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera.hh"
# include "camera_pike.hh"
# include "camera_xiq.hh"
# include "gl_window.hh"
# include "recorder.hh"
# include "queue.hh"

# include <cassert>

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
    };

    Holovibes(enum camera_type c, unsigned int buffer_nb_elts);
    ~Holovibes();

    void init_display(
      unsigned int w,
      unsigned int h);
    void dispose_display();
    void update_display();

    void init_camera();
    void dispose_camera();

  private:
    camera::Camera* camera_;
    GLWindow gl_window_;
  };
}

#endif /* !HOLOVIBES_HH */
