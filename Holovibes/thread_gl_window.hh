#ifndef THREAD_GL_WINDOW_HH
# define THREAD_GL_WINDOW_HH

# include <thread>

# include "gl_window.hh"
# include "queue.hh"
# include "frame_desc.hh"

namespace holovibes
{
  class ThreadGLWindow
  {
  public:
    ThreadGLWindow(
      Queue& queue_,
      const camera::FrameDescriptor& frame_desc,
      const char* title,
      int width,
      int height);
    ~ThreadGLWindow();

  private:
    void thread_proc();

  private:
    Queue& queue_;
    const camera::FrameDescriptor& frame_desc_;

    const char* title_;
    const int width_;
    const int height_;

    bool stop_requested_;
    std::thread thread_;
  };
}

#endif /* !THREAD_GL_WINDOW_HH */