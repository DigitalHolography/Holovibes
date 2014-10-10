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
    , gl_window_()
  {
    try
    {
      if (c == IDS)
        camera_ = new camera::CameraIds();
      else if (c == PIKE)
        camera_ = new camera::CameraPike();
      else if (c == XIQ)
        camera_ = new camera::CameraXiq();
      else
        assert(!"Impossible case");

      if (!camera_)
        throw std::exception("Error while allocating Camera constructor");

      const camera::s_frame_desc& desc = camera_->get_frame_descriptor();
      queue_ = new Queue(desc.get_frame_size(), buffer_nb_elts);
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
    delete camera_;
    delete queue_;
  }

  void Holovibes::init_display(
    unsigned int w,
    unsigned int h)
  {
    gl_window_.wnd_register_class();
    gl_window_.wnd_init("Holovibes", w, h);
    gl_window_.gl_init();
    gl_window_.gl_enable(w, h);
    gl_window_.wnd_show();
  }

  void Holovibes::dispose_display()
  {
    gl_window_.gl_disable();
    gl_window_.gl_free();
    gl_window_.wnd_unregister_class();
  }

  void Holovibes::update_display()
  {
    const camera::s_frame_desc& desc = camera_->get_frame_descriptor();

    queue_->enqueue(camera_->get_frame());

    gl_window_.gl_draw(queue_->dequeue(), desc.width, desc.height);
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
