#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera.hh"
# include "thread_compute.hh"
# include "thread_gl_window.hh"
# include "thread_capture.hh"
# include "recorder.hh"
# include "compute_descriptor.hh"
# include "pipeline.hh"

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

    Pipeline& init_compute(ComputeDescriptor& desc);
    void dispose_compute();

  private:
    camera::Camera* camera_;
    ThreadCapture* tcapture_;
    ThreadCompute* tcompute_;
    ThreadGLWindow* tglwnd_;
    Recorder* recorder_;

    Queue* input_;
    Queue* output_;
  };
}

#endif /* !HOLOVIBES_HH */
