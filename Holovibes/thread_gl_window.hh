#ifndef THREAD_GL_WINDOW_HH
# define THREAD_GL_WINDOW_HH

# include <thread>

# include "gl_window.hh"
# include "camera.hh"

namespace holovibes
{
  class ThreadGLWindow
  {
  public:
    // TODO: use Queue instead of Camera.
    ThreadGLWindow(
      camera::Camera& camera,
      const char* title,
      int width,
      int height);
    ~ThreadGLWindow();

  private:
    void thread_proc();
    
  private:
    camera::Camera& camera_;

    const char* title_;
    const int width_;
    const int height_;

    bool stop_requested_;
    std::thread thread_;
  };
}

#endif /* !THREAD_GL_WINDOW_HH */