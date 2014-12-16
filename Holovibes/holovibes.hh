#ifndef HOLOVIBES_HH
# define HOLOVIBES_HH

# include <icamera.hh>
# include "thread_compute.hh"
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
      return camera_ != nullptr;
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

    const std::string get_camera_ini_path() const
    {
      return camera_->get_ini_path();
    }

    std::vector<std::tuple<float, float, float>>& get_average_vector()
    {
      return average_vector_;
    }

  private:
    camera::ICamera* camera_;
    ThreadCapture* tcapture_;
    ThreadCompute* tcompute_;
    Recorder* recorder_;

    Queue* input_;
    Queue* output_;
    Pipeline* pipeline_;
    ComputeDescriptor compute_desc_;

    std::vector<std::tuple<float, float, float>> average_vector_;
  };
}

#endif /* !HOLOVIBES_HH */
