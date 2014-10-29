#include "stdafx.h"
#include "thread_gl_window.hh"
#include "gl_component.hh"
#include "tools.cuh"
#include <chrono>

#define GLWINDOW_FPS 30

namespace holovibes
{
  ThreadGLWindow::ThreadGLWindow(
    Queue& queue,
    const camera::FrameDescriptor& frame_desc,
    const char* title,
    int width,
    int height)
    : queue_(queue)
    , frame_desc_(frame_desc)
    , title_(title)
    , width_(width)
    , height_(height)
    , stop_requested_(false)
    , thread_(&ThreadGLWindow::thread_proc, this)
  {}

  ThreadGLWindow::~ThreadGLWindow()
  {
    stop_requested_ = true;
    if (thread_.joinable())
      thread_.join();
  }

  void ThreadGLWindow::thread_proc()
  {
    GLWindow glw(title_, width_, height_);
    glw.wnd_show();

    while (glw.running())
    {
      if (stop_requested_)
        glw.send_wm_close();
      glw.wnd_msgs_handler();
      GLComponent& gl = glw.get_gl_component();

      // FIXME
      // Temporary solution: gl thread gets image from
      // the queue (GPU) and copy it in CPU to display it. The display
      // has to be fetch directly in GPU in the future (no copies).
      //unsigned short* shifted = (unsigned short*)queue_.get_last_images(1);
      //shift_corners(&shifted, queue_.get_frame_desc().width, queue_.get_frame_desc().height);
      void* shifted = queue_.get_last_images(1);
      void* frame = malloc(queue_.get_size());
      cudaMemcpy(frame, shifted, queue_.get_size(), cudaMemcpyDeviceToHost);

      gl.gl_draw(frame, frame_desc_);
      free(frame);
      std::this_thread::sleep_for(
        std::chrono::milliseconds(1000 / GLWINDOW_FPS));
    }
  }
}