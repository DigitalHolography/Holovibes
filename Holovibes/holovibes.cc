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
    , tglwnd_(nullptr)
    , tcapture_(nullptr)
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
      throw std::exception("Error while allocating Camera constructor");
  }

  Holovibes::~Holovibes()
  {
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
    const camera::FrameDescriptor& desc = camera_->get_frame_descriptor();
    Queue& queue = tcapture_->get_queue();
    tglwnd_ = new ThreadGLWindow(queue, desc, "OpenGL", width, height);
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
  }

  void Holovibes::dispose_capture()
  {
    delete tcapture_;
    tcapture_ = nullptr;
    camera_->stop_acquisition();
    camera_->shutdown_camera();
  }
}
