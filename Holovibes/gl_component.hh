#ifndef GL_COMPONENT_HH
# define GL_COMPONENT_HH

# include <GL/GL.h>
# include <Windows.h>

# include "frame_desc.hh"

namespace holovibes
{
  /*! Contains a GL component to use in Win32 API window.
   * His purpose is to draw a frame on the window.
   */
  class GLComponent
  {
  public:
    GLComponent(HWND hwnd, int width, int height);
    ~GLComponent();

    /*! Draw a frame. */
    void gl_draw(
      const void* frame,
      const camera::s_frame_desc& desc);

  private:
    /*! Initialize the OpenGL PixelFormatDescriptor.
     * PFD informs the system how we are going to use the DC.
     */
    PIXELFORMATDESCRIPTOR get_pfd();
    /*! Enable OpenGL features and set the viewport. */
    void gl_enable(int width, int height);
    void gl_disable();

  private:
    /*! Window handle */
    HWND hwnd_;
    /*! Device context: permits to draw on the window. */
    HDC hdc_;
    /*! OpenGL render context. This is our bridge to the OpenGL system. */
    HGLRC hrc_;

    /*! The frame to draw is stored in a GLTexture2D. */
    GLuint texture_;
  };
}

#endif /* !GL_COMPONENT_HH */