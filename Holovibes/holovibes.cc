#include "holovibes.hh"
#include <frame_desc.hh>

#include <exception>
#include <cassert>
#include <memory>
#include <icamera.hh>

namespace holovibes
{
  Holovibes::Holovibes()
    : camera_loader_()
    , tcapture_(nullptr)
    , tcompute_(nullptr)
    , recorder_(nullptr)
    , input_(nullptr)
    , output_(nullptr)
    , pipeline_(nullptr)
    , compute_desc_()
    , average_queue_()
  {
  }

  Holovibes::~Holovibes()
  {
    delete tcompute_;
    delete tcapture_;
    delete input_;
    delete output_;
  }

  void Holovibes::init_capture(enum camera_type c, unsigned int buffer_nb_elts)
  {
    try
    {
      if (c == EDGE)
        camera_loader_.load_camera("CameraPCOEdge.dll");
      else if (c == IDS)
        camera_loader_.load_camera("CameraIds.dll");
      else if (c == IXON)
        camera_loader_.load_camera("CameraIxon.dll");
      else if (c == PIKE)
        camera_loader_.load_camera("CameraPike.dll");
      else if (c == PIXELFLY)
        camera_loader_.load_camera("CameraPCOPixelfly.dll");
      else if (c == XIQ)
        camera_loader_.load_camera("CameraXiq.dll");
      else
        assert(!"Impossible case");

      std::unique_ptr<camera::ICamera>& camera = camera_loader_.get_camera();
      camera->init_camera();
      input_ = new Queue(camera->get_frame_descriptor(), buffer_nb_elts);
      camera->start_acquisition();
      tcapture_ = new ThreadCapture(*camera, *input_);
      std::cout << "[CAPTURE] capture thread started" << std::endl;
    }
    catch (std::exception& e)
    {
      delete tcapture_;
      tcapture_ = nullptr;
      delete input_;
      input_ = nullptr;
      camera_loader_.get_camera().reset(nullptr);
      // Do NOT unload the library (because of the _re_throw)
      throw;
    }
  }

  void Holovibes::dispose_capture()
  {
    delete tcapture_;
    tcapture_ = nullptr;

    std::unique_ptr<camera::ICamera>& camera = camera_loader_.get_camera();

    if (camera)
    {
      camera->stop_acquisition();
      camera->shutdown_camera();
    }

    delete input_;
    input_ = nullptr;
    camera_loader_.unload_camera();

    std::cout << "[CAPTURE] capture thread stopped" << std::endl;
  }

  void Holovibes::init_recorder(
    std::string& filepath,
    unsigned int rec_n_images)
  {
    assert(camera_loader_.get_camera() && "camera not initialized");
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
    assert(camera_loader_.get_camera() && "camera not initialized");
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
