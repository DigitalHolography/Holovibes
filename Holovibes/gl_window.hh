#ifndef GL_WINDOW_HH
# define GL_WINDOW_HH

# include <Windows.h>
# include <GL/GL.h>

namespace holovibes
{
  class GLWindow
  {
  public:
    GLWindow()
      : hinstance_(GetModuleHandle(NULL))
      , hwnd_(nullptr)
      , hdc_(nullptr)
      , hrc_(nullptr)
      , texture_(0)
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
    /* OpenGL configuration. */
    void gl_enable(int width, int height);
    void gl_disable();
    /* Draw a frame. */
    void gl_draw(
      void* frame,
      int frame_width,
      int frame_height);
    /* Unload OpenGL ressources. */
    void gl_free();

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

    GLuint texture_;
  };
}

#endif /* !GL_WINDOW_HH */
