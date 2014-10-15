#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera.hh"
# include "camera_ids.hh"
# include "camera_pike.hh"
# include "camera_xiq.hh"
# include "camera_ids.hh"
# include "thread_gl_window.hh"
# include "recorder.hh"
# include "queue.hh"

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

    Holovibes(enum camera_type c, unsigned int buffer_nb_elts);
    ~Holovibes();

    void init_display(
      unsigned int w,
      unsigned int h);
    void dispose_display();

    void init_camera();
    void dispose_camera();

    camera::Camera* get_camera()
    {
      return camera_;
    }

    Queue* get_queue()
    {
      return queue_;
    }

  private:
    camera::Camera* camera_;
    ThreadGLWindow* tglhwnd_;
    Queue* queue_;
    Recorder* rec_;
  };
}

#endif /* !HOLOVIBES_HH */
