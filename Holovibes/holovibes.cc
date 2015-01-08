#include "holovibes.hh"
#include <frame_desc.hh>

#include <exception>
#include <cassert>
#include <memory>
#include <icamera.hh>

namespace holovibes
{
  Holovibes::Holovibes()
    : camera_()
    , camera_initialized_(false)
    , tcapture_()
    , tcompute_()
    , recorder_()
    , input_()
    , output_()
    , compute_desc_()
    , average_queue_()
  {
  }

  Holovibes::~Holovibes()
  {}

  void Holovibes::init_capture(enum camera_type c, unsigned int buffer_nb_elts)
  {
    camera_initialized_ = false;

    try
    {
      if (c == EDGE)
        camera_ = camera::CameraDLL::load_camera("CameraPCOEdge.dll");
      else if (c == IDS)
        camera_ = camera::CameraDLL::load_camera("CameraIds.dll");
      else if (c == IXON)
        camera_ = camera::CameraDLL::load_camera("CameraIxon.dll");
      else if (c == PIKE)
        camera_ = camera::CameraDLL::load_camera("CameraPike.dll");
      else if (c == PIXELFLY)
        camera_ = camera::CameraDLL::load_camera("CameraPCOPixelfly.dll");
      else if (c == XIQ)
        camera_ = camera::CameraDLL::load_camera("CameraXiq.dll");
      else
        assert(!"Impossible case");

      camera_->init_camera();
      input_.reset(new Queue(camera_->get_frame_descriptor(), buffer_nb_elts));
      camera_->start_acquisition();
      tcapture_.reset(new ThreadCapture(*camera_, *input_));
      std::cout << "[CAPTURE] capture thread started" << std::endl;
      camera_initialized_ = true;
    }
    catch (std::exception& e)
    {
      tcapture_.reset(nullptr);
      input_.reset(nullptr);

      throw;
    }
  }

  void Holovibes::dispose_capture()
  {
    tcapture_.reset(nullptr);
    if (camera_ && camera_initialized_)
    {
      camera_->stop_acquisition();
      camera_->shutdown_camera();
    }

    input_.reset(nullptr);
    camera_.reset();
    camera_initialized_ = false;

    std::cout << "[CAPTURE] capture thread stopped" << std::endl;
  }

  void Holovibes::init_recorder(
    std::string& filepath,
    unsigned int rec_n_images)
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    if (tcompute_)
    {
      recorder_.reset(new Recorder(*output_, filepath));
    }
    else
    {
      recorder_.reset(new Recorder(*input_, filepath));
    }
    std::cout << "[RECORDER] recorder initialized" << std::endl;
    recorder_->record(rec_n_images);
  }

  void Holovibes::dispose_recorder()
  {
    recorder_.reset(nullptr);
  }

  void Holovibes::init_compute()
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    assert(input_ && "input queue not initialized");

    camera::FrameDescriptor output_frame_desc = input_->get_frame_desc();
    output_frame_desc.depth = 2;
    output_.reset(new Queue(output_frame_desc, input_->get_max_elts()));

    tcompute_.reset(new ThreadCompute(compute_desc_, *input_, *output_));
    std::cout << "[CUDA] compute thread started" << std::endl;

    // A wait_for is necessary here in order for the pipeline to finish
    // its allocations before getting it.
    std::mutex mutex;
    std::unique_lock<std::mutex> lck(mutex);

    std::cout << "Pipeline is initializing ";
    while (tcompute_->get_memory_cv().wait_for(lck, std::chrono::milliseconds(100)) == std::cv_status::timeout)
      std::cout << ".";
    std::cout << "\n";
  }

  void Holovibes::dispose_compute()
  {
    tcompute_.reset(nullptr);
    output_.reset(nullptr);
  }
}
