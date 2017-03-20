#include <Windows.h>
#include "thread_capture.hh"
#include "info_manager.hh"
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
    gui::InfoManager::get_manager()->update_info_safe("ImgSource", camera_.get_name());
  }

  ThreadCapture::~ThreadCapture()
  {
    stop_requested_ = true;

    while (!thread_.joinable())
      continue;
    thread_.join();
    gui::InfoManager::get_manager()->update_info_safe("ImgSource", "None");
  }

  void ThreadCapture::thread_proc()
  {
	  SetThreadPriority(thread_.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
    while (!stop_requested_)
      queue_.enqueue(camera_.get_frame(), cudaMemcpyHostToDevice);
  }

  const camera::FrameDescriptor& ThreadCapture::get_frame_descriptor() const
  {
    return camera_.get_frame_descriptor();
  }
}
