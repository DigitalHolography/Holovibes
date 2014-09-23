#ifndef GL_WINDOW_HH
# define GL_WINDOW_HH

# include <Windows.h>
# include <GL/GL.h>

namespace gui
{
  class GLWindow
  {
  public:
    GLWindow()
      : hinstance_(GetModuleHandle(NULL))
      , hwnd_(nullptr)
      , hdc_(nullptr)
      , hrc_(nullptr)
    {}
    ~GLWindow()
    {}

    /* Register the Window Class. */
    bool wnd_class_register();
    /* Initialize the window. */
    bool wnd_init(
      const char* title,
      int width,
      int height);
    /* Show the window. */
    void wnd_show();
    /* Initialize OpenGL. */
    void gl_init();

    const HDC& get_hdc() const
    {
      return hdc_;
    }

  private:
    /* Copy is not allowed. */
    GLWindow(const GLWindow&)
    {}
    GLWindow& operator=(const GLWindow&)
    {}

    /* Retrieve the PFD. */
    PIXELFORMATDESCRIPTOR gl_get_pfd();

    static LRESULT CALLBACK wnd_proc(
      HWND hwnd,
      UINT msg,
      WPARAM wparam,
      LPARAM lparam);

  private:
    static bool class_registered_;

    HINSTANCE hinstance_;
    HWND hwnd_;
    HDC hdc_;
    HGLRC hrc_;
  };
}

#endif /* !GL_WINDOW_HH */
