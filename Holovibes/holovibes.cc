#include "holovibes.hh"
#include "frame_desc.hh"
#include "gl_component.hh"
#include "camera_ids.hh"
#include "camera_pike.hh"
#include "camera_pixelfly.hh"
#include "camera_xiq.hh"

#include <exception>
#include <cassert>

namespace holovibes
{
  Holovibes::Holovibes()
    : camera_(nullptr)
    , tcapture_(nullptr)
    , tcompute_(nullptr)
    , recorder_(nullptr)
    , input_(nullptr)
    , output_(nullptr)
    , pipeline_(nullptr)
    , compute_desc_()
  {
  }

  Holovibes::~Holovibes()
  {
    delete tcompute_;
    delete tcapture_;
    delete camera_;
    delete input_;
    delete output_;
  }

  void Holovibes::init_display(
    unsigned int width,
    unsigned int height)
  {
  }

  void Holovibes::dispose_display()
  {
  }

  void Holovibes::init_capture(enum camera_type c, unsigned int buffer_nb_elts)
  {
    if (c == IDS)
      camera_ = new camera::CameraIds();
    else if (c == PIKE)
      camera_ = new camera::CameraPike();
    else if (c == PIXELFLY)
      camera_ = new camera::CameraPixelfly();
    else if (c == XIQ)
      camera_ = new camera::CameraXiq();
    else
      assert(!"Impossible case");

    if (!camera_)
      throw std::runtime_error("Error while allocating Camera constructor");

    assert(camera_ && "camera not initialized");
    camera_->init_camera();
    input_ = new Queue(camera_->get_frame_descriptor(), buffer_nb_elts);
    camera_->start_acquisition();
    tcapture_ = new ThreadCapture(*camera_, *input_);
    std::cout << "[CAPTURE] capture thread started" << std::endl;
  }

  void Holovibes::dispose_capture()
  {
    delete tcapture_;
    tcapture_ = nullptr;
    camera_->stop_acquisition();
    camera_->shutdown_camera();
    delete input_;
    input_ = nullptr;
  }

  void Holovibes::init_recorder(
    std::string& filepath,
    unsigned int rec_n_images)
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    if (tcompute_)
    {
      recorder_ = new Recorder(*output_, filepath);
    }
    else
    {
      recorder_ = new Recorder(*input_, filepath);
    }
    std::cout << "[RECORDER] recorder initialized" << std::endl;
    recorder_->record(rec_n_images);
  }

  void Holovibes::dispose_recorder()
  {
    delete recorder_;
    recorder_ = nullptr;
  }

  void Holovibes::init_compute()
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    assert(input_ && "input queue not initialized");

    camera::FrameDescriptor output_frame_desc = input_->get_frame_desc();
    output_frame_desc.depth = 2;
    output_ = new Queue(output_frame_desc, input_->get_max_elts());

    tcompute_ = new ThreadCompute(compute_desc_, *input_, *output_);
    std::cout << "[CUDA] compute thread started" << std::endl;

    // A wait_for is necessary here in order for the pipeline to finish
    // its allocations before getting it.
    std::mutex mutex;
    std::unique_lock<std::mutex> lck(mutex);

    std::cout << "Pipeline is initializing ";
    while (tcompute_->get_memory_cv().wait_for(lck, std::chrono::milliseconds(100)) == std::cv_status::timeout)
      std::cout << ".";
    std::cout << "\n";

    pipeline_ = &tcompute_->get_pipeline();
  }

  void Holovibes::dispose_compute()
  {
    delete tcompute_;
    tcompute_ = nullptr;
    pipeline_ = nullptr;
    delete output_;
    output_ = nullptr;
  }
}
