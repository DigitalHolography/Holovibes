#include "stdafx.h"
#include "gl_component.hh"

namespace holovibes
{
  GLComponent::GLComponent(HWND hwnd, int width, int height)
    : hwnd_(hwnd)
    , hdc_(GetDC(hwnd))
    , hrc_(nullptr)
    , texture_(0)
  {
    PIXELFORMATDESCRIPTOR pfd = get_pfd();

    const int pixel_format = ChoosePixelFormat(hdc_, &pfd);
    if (!pixel_format)
      throw std::exception("unable to find a suitable pixel format");

    if (!SetPixelFormat(hdc_, pixel_format, &pfd))
      throw std::exception("can not set the pixel format");

    hrc_ = wglCreateContext(hdc_);

    if (!hrc_)
      throw std::exception("unable to create GL context");

    if (!wglMakeCurrent(hdc_, hrc_))
      throw std::exception("unable to make current GL context");

    gl_enable(width, height);
  }

  GLComponent::~GLComponent()
  {
    gl_disable();
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hrc_);
    ReleaseDC(hwnd_, hdc_);
  }

  inline PIXELFORMATDESCRIPTOR GLComponent::get_pfd()
  {
    PIXELFORMATDESCRIPTOR pfd;

    /* Init with null values. */
    memset(&pfd, 0, sizeof (PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof (PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;

    return pfd;
  }

  /*! \brief OpenGL configuration.
  * \param width Width of GL viewport.
  * \param height Height of GL viewport.
  */
  void GLComponent::gl_enable(
    int width,
    int height)
  {
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_QUADS);

    glGenTextures(1, &texture_);
    glViewport(0, 0, width, height);
  }

  void GLComponent::gl_disable()
  {
    glDisable(GL_QUADS);
    glDisable(GL_TEXTURE_2D);
  }

  void GLComponent::gl_draw(
    const void* frame,
    const camera::FrameDescriptor& desc)
  {
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (desc.endianness == camera::BIG_ENDIAN)
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
    else
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);

    glTexImage2D(
      GL_TEXTURE_2D,
      /* Base image level. */
      0,
      GL_LUMINANCE,
      desc.width,
      desc.height,
      /* border: This value must be 0. */
      0,
      GL_LUMINANCE,
      /* Unsigned byte = 1 byte, Unsigned short = 2 bytes. */
      desc.depth == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
      /* Pointer to image data in memory. */
      frame);

    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0); glVertex2d(-1.0, +1.0);
    glTexCoord2d(1.0, 0.0); glVertex2d(+1.0, +1.0);
    glTexCoord2d(1.0, 1.0); glVertex2d(+1.0, -1.0);
    glTexCoord2d(0.0, 1.0); glVertex2d(-1.0, -1.0);
    glEnd();

    SwapBuffers(hdc_);
    glDeleteTextures(1, &texture_);
  }
}