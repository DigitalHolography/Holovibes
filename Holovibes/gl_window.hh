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
    /*! \brief Constructor of GLWindow object.
     * It initialize object's fields.
    **/
    GLWindow()
      : hinstance_(GetModuleHandle(NULL))
      , hwnd_(nullptr)
      , hdc_(nullptr)
      , hrc_(nullptr)
      , texture_(0)
    {}
    /*! \brief Default destructor.
     */
    ~GLWindow()
    {}

    /*! \brief Register the Window Class for subsequent use in calls to the
     * CreateWindow function.
     * \remarks A window class can be registered only once, otherwise it will
     * generate a WGL_CLASS_REGISTERED error.
     */
    bool wnd_class_register();
    /*! \brief Initialize the window.
     * \param title Title of the window.
     * \param width Width of the window.
     * \param height Height of the window.
     */
    bool wnd_init(
      const char* title,
      int width,
      int height);
    /* Show the window. */
    void wnd_show();
    /* Initialize OpenGL. */
    void gl_init();
    /*! \brief OpenGL configuration.
     * \param width Width of GL viewport.
     * \param height Height of GL viewport.
     */
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
