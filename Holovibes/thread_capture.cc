#include "thread_capture.hh"
#include "icamera.hh"
#include "queue.hh"

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

    while (!thread_.joinable())
      continue;
    thread_.join();
  }

  void ThreadCapture::thread_proc()
  {
    while (!stop_requested_)
      queue_.enqueue(camera_.get_frame(), cudaMemcpyHostToDevice);
  }
}