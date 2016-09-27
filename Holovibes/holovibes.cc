#include <exception>
#include <cassert>
#include <memory>
#include <boost/filesystem.hpp>

#include "holovibes.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "icamera.hh"
#include "config.hh"
#include "camera_dll.hh"

namespace holovibes
{
  Holovibes::Holovibes()
    : camera_()
    , camera_initialized_(false)
    , tcapture_()
    , tcompute_()
    , input_()
    , output_()
    , compute_desc_()
    , average_queue_()
    , launch_path(boost::filesystem::current_path().generic_string())
  {
  }

  Holovibes::~Holovibes()
  {
  }

  void Holovibes::init_capture(const enum camera_type c)
  {
    camera_initialized_ = false;
    try
    {
      if (c == ADIMEC)
        camera_ = camera::CameraDLL::load_camera("CameraAdimec.dll");
      else if (c == EDGE)
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

      std::cout << "(Holovibes) Prepared to initialize camera." << std::endl;
      camera_->init_camera();
      std::cout << "(Holovibes) Prepared to reset queues." << std::endl;
      input_.reset(new Queue(camera_->get_frame_descriptor(), global::global_config.input_queue_max_size, "InputQueue"));
      std::cout << "(Holovibes) Prepared to start initialization." << std::endl;
      camera_->start_acquisition();
      tcapture_.reset(new ThreadCapture(*camera_, *input_));

      std::cout << "[CAPTURE] capture thread started" << std::endl;
      camera_initialized_ = true;
    }
    catch (std::exception& e)
    {
      std::cout << e.what() << std::endl;
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

  void Holovibes::recorder(
    const std::string& filepath,
    const unsigned int rec_n_images)
  {
    Recorder* recorder;

    assert(camera_initialized_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");

    if (tcompute_)
      recorder = new Recorder(*output_, filepath);
    else
      recorder = new Recorder(*input_, filepath);

    std::cout << "[RECORDER] recorder Start" << std::endl;
    recorder->record(rec_n_images);
    delete recorder;
    std::cout << "[RECORDER] recorder Stop" << std::endl;
  }

  void Holovibes::init_compute(const ThreadCompute::PipeType pipetype)
  {
    assert(camera_initialized_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    assert(input_ && "input queue not initialized");

    camera::FrameDescriptor output_frame_desc = input_->get_frame_desc();
    output_frame_desc.depth = 2;
    output_.reset(new Queue(output_frame_desc, global::global_config.output_queue_max_size, "OutputQueue"));

    tcompute_.reset(new ThreadCompute(compute_desc_, *input_, *output_, pipetype));
    std::cout << "[CUDA] compute thread started" << std::endl;

    // A wait_for is necessary here in order for the pipe to finish
    // its allocations before getting it.
    std::mutex mutex;
    std::unique_lock<std::mutex> lck(mutex);

    std::cout << "Pipe is initializing ";
    while (tcompute_->get_memory_cv().wait_for(lck, std::chrono::milliseconds(100)) == std::cv_status::timeout)
      std::cout << ".";
    std::cout << "\n";
  }

  void Holovibes::dispose_compute()
  {
    tcompute_.reset(nullptr);
    output_.reset(nullptr);
  }

  void Holovibes::reset_convolution_matrix()
  {
	  compute_desc_.convo_matrix_length = 0;
	  compute_desc_.convo_matrix_width = 0;
	  compute_desc_.convo_matrix_z = 0;
	  compute_desc_.convo_matrix.clear();
  }

  const camera::FrameDescriptor& Holovibes::get_cam_frame_desc()
  {
    return (tcapture_.get()->get_frame_descriptor());
  }

  const float Holovibes::get_boundary()
  {
    const float n = static_cast<float>(get_cam_frame_desc().height);
    const float d = get_cam_frame_desc().pixel_size * static_cast<float>(0.000001);
    return (n * d * d) / compute_desc_.lambda;
  }

  void Holovibes::init_import_mode(std::string &file_src,
    camera::FrameDescriptor frame_desc,
    bool loop,
    unsigned int fps,
    unsigned int spanStart,
    unsigned int spanEnd,
    unsigned int q_max_size_)
  {
    camera_initialized_ = false;

    try
    {
      input_.reset(new Queue(frame_desc, q_max_size_, "InputQueue"));
      tcapture_.reset(
        new ThreadReader(file_src
        , frame_desc
        , loop
        , fps
        , spanStart
        , spanEnd
        , *input_));
      std::cout << "[CAPTURE] reader thread started" << std::endl;
      camera_initialized_ = true;
    }
    catch (std::exception& e)
    {
      std::cout << e.what() << std::endl;
      tcapture_.reset(nullptr);
      input_.reset(nullptr);

      throw;
    }
  }

  const std::string Holovibes::get_launch_path()
  {
    return launch_path;
  }
}