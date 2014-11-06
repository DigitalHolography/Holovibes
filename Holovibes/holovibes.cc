#include "stdafx.h"
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
  Holovibes::Holovibes(enum camera_type c)
    : camera_(nullptr)
    , tcapture_(nullptr)
    , tcompute_(nullptr)
    , tglwnd_(nullptr)
    , recorder_(nullptr)
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
  }

  Holovibes::~Holovibes()
  {
    delete tcompute_;
    delete tcapture_;
    delete tglwnd_;
    delete camera_;
  }

  void Holovibes::init_display(
    unsigned int width,
    unsigned int height)
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");

    if (tcompute_)
    {
      Queue& queue = tcompute_->get_queue();
      tglwnd_ = new ThreadGLWindow(queue, "OpenGL", width, height);
    }
    else
    {
      Queue& queue = tcapture_->get_queue();
      tglwnd_ = new ThreadGLWindow(queue, "OpenGL", width, height);
    }
    std::cout << "[DISPLAY] display thread started" << std::endl;
  }

  void Holovibes::dispose_display()
  {
    delete tglwnd_;
    tglwnd_ = nullptr;
  }

  void Holovibes::init_capture(unsigned int buffer_nb_elts)
  {
    assert(camera_ && "camera not initialized");
    camera_->init_camera();
    camera_->start_acquisition();
    tcapture_ = new ThreadCapture(*camera_, buffer_nb_elts);
    std::cout << "[CAPTURE] capture thread started" << std::endl;
  }

  void Holovibes::dispose_capture()
  {
    delete tcapture_;
    tcapture_ = nullptr;
    camera_->stop_acquisition();
    camera_->shutdown_camera();
  }

  void Holovibes::init_recorder(
    std::string& filepath,
    unsigned int rec_n_images)
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    if (tcompute_)
    {
      recorder_ = new Recorder(tcompute_->get_queue(), filepath);
    }
    else
    {
      recorder_ = new Recorder(tcapture_->get_queue(), filepath);
    }
    std::cout << "[RECORDER] recorder initialized" << std::endl;
    recorder_->record(rec_n_images);
  }

  void Holovibes::dispose_recorder()
  {
    delete recorder_;
    recorder_ = nullptr;
  }

  void Holovibes::init_compute(ComputeDescriptor& desc)
  {
    assert(camera_ && "camera not initialized");
    assert(tcapture_ && "capture thread not initialized");
    tcompute_ = new ThreadCompute(desc, tcapture_->get_queue());
  }

  void Holovibes::dispose_compute()
  {
    delete tcompute_;
    tcompute_ = nullptr;
  }
}
