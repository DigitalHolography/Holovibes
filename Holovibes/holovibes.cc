#include "stdafx.h"
#include "holovibes.hh"
#include "frame_desc.hh"

#include <exception>

namespace holovibes
{
  Holovibes::Holovibes(
    enum camera_type c,
    unsigned int buffer_nb_elts)
    : camera_(nullptr)
    , window_(nullptr)
  {
    try
    {
      if (c == PIKE)
        camera_ = new camera::CameraPike();
      else if (c == XIQ)
        camera_ = new camera::CameraXiq();
      else if (c == IDS)
        camera_ = new camera::CameraIds();
      else
        assert(!"Impossible case");

      if (!camera_)
        throw std::exception("Error while allocating Camera constructor");
#if 0
      const camera::s_frame_desc& desc = camera_->get_frame_descriptor();
      frames_ = Queue(desc.get_frame_size(), buffer_nb_elts);
#endif
    }
    catch (...)
    {
      if (camera_)
        delete camera_;

      // Throw the exception again, without memory leak.
      throw;
    }
  }

  Holovibes::~Holovibes()
  {
    if (camera_)
      delete camera_;
    if (window_)
      delete window_;
  }

  void Holovibes::init_display(
    unsigned int width,
    unsigned int height)
  {
    window_ = new GLWindow("OpenGL", width, height);
    window_->wnd_show();
  }

  void Holovibes::dispose_display()
  {
    if (window_)
    {
      delete window_;
      window_ = nullptr;
    }
  }

  void Holovibes::update_display()
  {
    const camera::s_frame_desc& desc = camera_->get_frame_descriptor();
    window_->gl_draw(camera_->get_frame(), camera_->get_frame_descriptor());
  }

  void Holovibes::init_camera()
  {
    camera_->init_camera();
    camera_->start_acquisition();
  }

  void Holovibes::dispose_camera()
  {
    camera_->stop_acquisition();
    camera_->shutdown_camera();
  }
}
