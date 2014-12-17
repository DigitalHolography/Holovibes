#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include "camera_loader.hh"
# include "thread_compute.hh"
# include "thread_capture.hh"
# include "recorder.hh"
# include "compute_descriptor.hh"
# include "pipeline.hh"
# include "concurrent_deque.hh"

namespace holovibes
{
  class Holovibes
  {
  public:
    enum camera_type
    {
      NONE,
      EDGE,
      IDS,
      IXON,
      PIKE,
      PIXELFLY,
      XIQ,
    };

    Holovibes();
    ~Holovibes();

    void init_capture(enum camera_type c, unsigned int buffer_nb_elts);
    void dispose_capture();

    bool is_camera_initialized()
    {
      return camera_loader_.get_camera().operator bool();
    }

    Queue& get_capture_queue()
    {
      return *input_;
    }

    Queue& get_output_queue()
    {
      return *output_;
    }

    void init_recorder(
      std::string& filepath,
      unsigned int rec_n_images);
    void dispose_recorder();

    void init_compute();
    void dispose_compute();

    Pipeline& get_pipeline()
    {
      if (pipeline_)
        return *pipeline_;
      throw std::runtime_error("Pipeline is null");
    }

    ComputeDescriptor& get_compute_desc()
    {
      return compute_desc_;
    }

    void set_compute_desc(ComputeDescriptor& compute_desc)
    {
      compute_desc_ = compute_desc;
    }

    const char* get_camera_ini_path() const
    {
      return camera_loader_.get_camera()->get_ini_path();
    }

    ConcurrentDeque<std::tuple<float, float, float>>& get_average_queue()
    {
      return average_queue_;
    }

  private:
    camera::CameraLoader camera_loader_;
    ThreadCapture* tcapture_;
    ThreadCompute* tcompute_;
    Recorder* recorder_;

    Queue* input_;
    Queue* output_;
    Pipeline* pipeline_;
    ComputeDescriptor compute_desc_;

    ConcurrentDeque<std::tuple<float, float, float>> average_queue_;
  };
}

#endif /* !HOLOVIBES_HH */
