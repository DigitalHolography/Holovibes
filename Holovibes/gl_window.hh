#ifndef GL_WINDOW_HH
# define GL_WINDOW_HH

# include <Windows.h>
# include <GL/GL.h>

namespace holovibes
{
  /*! GLWindow class use Win32 API to display a window containing an OpenGL
   * widget.
   */
  class GLWindow
  {
  public:
    GLWindow(
      const char* title,
      int width,
      int height)
      : hinstance_(GetModuleHandle(NULL))
      , hwnd_(nullptr)
      , hdc_(nullptr)
      , hrc_(nullptr)
      , texture_(0)
    {
      if (!class_registered_)
        wnd_register_class();

      wnd_init(title, width, height);
      wnd_gl_init();
      gl_enable(width, height);
    }

    virtual ~GLWindow()
    {}

    /* Show the window. */
    void wnd_show();
    /*! \brief OpenGL configuration.
     * \param width Width of GL viewport.
     * \param height Height of GL viewport.
     */
    /* Draw a frame. */
    void gl_draw(
      void* frame,
      int frame_width,
      int frame_height);

  private:
    /*! \brief Register the Window Class for subsequent use in calls to the
     * CreateWindow function.
     * \remarks A window class can be registered only once, otherwise it will
     * generate a WGL_CLASS_REGISTERED error.
     */
    void wnd_register_class();

    /*! \brief Unregister the Window Class.
    */
    void wnd_unregister_class();

    /*! \brief Initialize the window.
     * \param title Title of the window.
     * \param width Width of the window.
     * \param height Height of the window.
     */
    void wnd_init(
      const char* title,
      int width,
      int height);

    /*! Retrieves the PFD. */
    PIXELFORMATDESCRIPTOR gl_get_pfd();

    /*! Initialize OpenGL in the window. */
    void wnd_gl_init();
    /*! Unload OpenGL ressources. */
    void wnd_gl_free();

    /*! Enable OpenGL features and set the viewport. */
    void gl_enable(int width, int height);
    void gl_disable();

    static LRESULT CALLBACK wnd_proc(
      HWND hwnd,
      UINT msg,
      WPARAM wparam,
      LPARAM lparam);

    /* Copy is not allowed. */
    GLWindow(const GLWindow&)
    {}
    GLWindow& operator=(const GLWindow&)
    {}

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
