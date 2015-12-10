#include "thread_capture.hh"

namespace holovibes
{
  ThreadCapture::ThreadCapture(
    camera::ICamera& camera,
    Queue& input)
    : IThreadInput()
    , camera_(camera)
    , queue_(input)
    , thread_(&ThreadCapture::thread_proc, this)
  {
  }

  ThreadCapture::~ThreadCapture()
  {
    stop_requested_ = true;

    if (thread_.joinable())
      thread_.join();
  }

  void ThreadCapture::thread_proc()
  {
    while (!stop_requested_)
      queue_.enqueue(camera_.get_frame(), cudaMemcpyHostToDevice);
  }
}