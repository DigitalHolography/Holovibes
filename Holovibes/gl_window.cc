#include "stdafx.h"
#include "gl_window.hh"
#include "error_handler.hh"

#include <cassert>
#include <exception>

namespace holovibes
{
#pragma region WIN32_WindowClass
  LRESULT GLWindow::wnd_proc(
    HWND hwnd,
    UINT msg,
    WPARAM wparam,
    LPARAM lparam)
  {
    return DefWindowProc(hwnd, msg, wparam, lparam);
  }

  void GLWindow::wnd_register_class()
  {
    /* Window class structure */
    WNDCLASS wc;

    wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = (WNDPROC)wnd_proc;
    wc.cbClsExtra = 0;
    wc.cbWndExtra = 0;
    wc.hInstance = hinstance_;
    wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = 0;
    wc.lpszMenuName = 0;
    wc.lpszClassName = "OpenGL";

    /* If RegisterClass fails, the return value is zero. */
    if (RegisterClass(&wc) == 0)
      throw std::exception("class is already registered");
  }

  void GLWindow::wnd_unregister_class()
  {
    UnregisterClass("OpenGL", hinstance_);
  }
#pragma endregion

#pragma region Window
  void GLWindow::wnd_init(
    const char* title,
    int width,
    int height)
  {
    hwnd_ = CreateWindow(
      "OpenGL",
      title,
      WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
      0, 0,
      width, height,
      NULL, NULL,
      hinstance_, NULL);

    if (!hwnd_)
      throw std::exception("failed to instanciate an OpenGL window class");
  }

  void GLWindow::wnd_show()
  {
    if (!hwnd_)
      throw std::exception("no window instanciated");

    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
  }
#pragma endregion

#pragma region OpenGL
  inline PIXELFORMATDESCRIPTOR GLWindow::gl_get_pfd()
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

  void GLWindow::wnd_gl_init()
  {
    /* Get the device context. */
    hdc_ = GetDC(hwnd_);

    PIXELFORMATDESCRIPTOR pfd = gl_get_pfd();
    int pixel_format = ChoosePixelFormat(hdc_, &pfd);
    if (!pixel_format)
      throw std::exception("unable to find a suitable pixel format");

    if (!SetPixelFormat(hdc_, pixel_format, &pfd))
      throw std::exception("can not set the pixel format");

    hrc_ = wglCreateContext(hdc_);

    if (!hrc_)
      throw std::exception("unable to create GL context");

    if (!wglMakeCurrent(hdc_, hrc_))
      throw std::exception("unable to make current GL context");
  }

  void GLWindow::gl_enable(
    int width,
    int height)
  {
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_QUADS);

    glGenTextures(1, &texture_);
    glViewport(0, 0, width, height);
  }

  void GLWindow::gl_draw(
    void* frame,
    const camera::s_frame_desc& desc)
  {
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if (desc.endianness == camera::BIG_ENDIAN)
      glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);

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
      desc.get_byte_depth() == 1 ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT,
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

  void GLWindow::gl_disable()
  {
    glDisable(GL_QUADS);
    glDisable(GL_TEXTURE_2D);
  }

  void GLWindow::wnd_gl_free()
  {
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hrc_);
    ReleaseDC(hwnd_, hdc_);
  }
#pragma endregion
}
