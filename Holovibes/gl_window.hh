#ifndef GL_WINDOW_HH
# define GL_WINDOW_HH

# include <Windows.h>
# include <GL/GL.h>

namespace gui
{
  class GLWindow
  {
  public:
    static GLWindow& get_instance()
    {
      if (!instance_)
        instance_ = new GLWindow();
      return *instance_;
    }

    static void shutdown()
    {
      if (instance_)
        delete instance_;
      instance_ = nullptr;
    }

    bool init(
      const char* title,
      int width,
      int height);

  private:
    GLWindow()
      : hinstance_(GetModuleHandle(NULL))
      , hwnd_(nullptr)
      , hdc_(nullptr)
      , hrc_(nullptr)
    {}
    GLWindow(const GLWindow&)
    {}
    GLWindow& operator=(const GLWindow&)
    {}
    ~GLWindow()
    {}

    static LRESULT CALLBACK wnd_proc(
      HWND hwnd,
      UINT msg,
      WPARAM wparam,
      LPARAM lparam);

  private:
    static GLWindow* instance_;

    HINSTANCE hinstance_;
    HWND hwnd_;
    HDC hdc_;
    HGLRC hrc_;
  };
}

#endif /* !GL_WINDOW_HH */
