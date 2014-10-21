#include "stdafx.h"
#include "thread_capture.hh"

namespace holovibes
{
  ThreadCapture::ThreadCapture(
    camera::Camera& camera,
    unsigned int buffer_nb_elts)
    : camera_(camera)
    , queue_(camera.get_frame_descriptor().frame_size(), buffer_nb_elts)
    , stop_requested_(false)
    , thread_(&ThreadCapture::thread_proc, this)
  {}

  ThreadCapture::~ThreadCapture()
  {
    stop_requested_ = true;
    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCapture::thread_proc()
  {
    while (!stop_requested_)
      queue_.enqueue(camera_.get_frame());
  }
}