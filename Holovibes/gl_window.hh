#ifndef GL_WINDOW_HH
# define GL_WINDOW_HH

# include <Windows.h>

# include "gl_component.hh"

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
      int height,
      const camera::FrameDescriptor& desc);

    virtual ~GLWindow();

    /* Show the window. */
    void wnd_show();

    GLComponent& get_gl_component()
    {
      return *gl_;
    }

    bool running() const
    {
      return running_;
    }

    void send_wm_close()
    {
      PostMessage(hwnd_, WM_CLOSE, 0, 0);
    }

    void wnd_msgs_handler();

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

    /*! Win32 API Message handler. */
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
    HINSTANCE hinstance_;
    HWND hwnd_;
    GLComponent* gl_;

    static bool running_;
  };
}

#endif /* !GL_WINDOW_HH */