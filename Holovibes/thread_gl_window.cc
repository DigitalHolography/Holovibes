#include "stdafx.h"
#include "thread_gl_window.hh"
#include "gl_component.hh"

#include <chrono>

#define GLWINDOW_FPS 30

namespace holovibes
{
  ThreadGLWindow::ThreadGLWindow(
    camera::Camera& camera,
    const char* title,
    int width,
    int height)
    : camera_(camera)
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
      gl.gl_draw(camera_.get_frame(), camera_.get_frame_descriptor());
      std::this_thread::sleep_for(
        std::chrono::milliseconds(1000 / GLWINDOW_FPS));
    }
  }
}