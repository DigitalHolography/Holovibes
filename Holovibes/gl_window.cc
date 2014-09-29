#include "stdafx.h"
#include "gl_window.hh"
#include "error_handler.hh"

#include <cassert>

namespace holovibes
{
  bool GLWindow::class_registered_ = false;

  bool GLWindow::wnd_register_class()
  {
    /* Init must be called only if Window Class
    ** has not been registered.
    */

    if (class_registered_)
    {
      assert("WndClass has already been registered.");
      ErrorHandler::get_instance()
        .send_error(WGL_CLASS_REGISTERED);

      return false;
    }

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
    class_registered_ = RegisterClass(&wc) != 0;
    return class_registered_;
  }

  void GLWindow::wnd_unregister_class()
  {
    if (class_registered_)
      UnregisterClass("OpenGL", hinstance_);

    class_registered_ = false;
  }

  bool GLWindow::wnd_init(
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
    {
      assert("Failed to initialize window");
      ErrorHandler::get_instance()
        .send_error(WGL_CREATE_WINDOW);
      return false;
    }

    return true;
  }

  void GLWindow::wnd_show()
  {
    if (!hwnd_)
    {
      assert("Window is not initialized");
      ErrorHandler::get_instance()
        .send_error(WGL_INIT_WINDOW);
      return;
    }
    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
  }

  inline PIXELFORMATDESCRIPTOR GLWindow::gl_get_pfd()
  {
    PIXELFORMATDESCRIPTOR pfd;

    /* Init with null values. */
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 16;

    return pfd;
  }

  void GLWindow::gl_init()
  {
    /* Get the device context. */
    hdc_ = GetDC(hwnd_);

    PIXELFORMATDESCRIPTOR pfd = gl_get_pfd();
    int pixel_format = ChoosePixelFormat(hdc_, &pfd);
    if (!pixel_format)
    {
      assert("Unable to find a suitable pixel format");
      ErrorHandler::get_instance()
        .send_error(WGL_PIXEL_FORMAT_FIND);
      return;
    }

    if (!SetPixelFormat(hdc_, pixel_format, &pfd))
    {
      assert("Cannot set the pixel format");
      ErrorHandler::get_instance()
        .send_error(WGL_PIXEL_FORMAT_SET);
      return;
    }

    hrc_ = wglCreateContext(hdc_);

    if (!hrc_)
    {
      assert("Unable to create GL context");
      ErrorHandler::get_instance()
        .send_error(WGL_GL_CONTEXT_CREATE);
      return;
    }

    if (!wglMakeCurrent(hdc_, hrc_))
    {
      assert("Unable to make current GL context");
      ErrorHandler::get_instance()
        .send_error(WGL_GL_CONTEXT_MAKECUR);
      return;
    }
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
    int frame_width,
    int frame_height)
  {
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(
      GL_TEXTURE_2D,
      /* Base image level. */
      0,
      GL_LUMINANCE,
      frame_width,
      frame_height,
      /* border: This value must be 0. */
      0,
      GL_LUMINANCE,
      /* Unsigned byte = 1 byte, Unsigned short = 2 bytes. */
      GL_UNSIGNED_BYTE,
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

  void GLWindow::gl_free()
  {
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(hrc_);
    ReleaseDC(hwnd_, hdc_);
  }

  LRESULT GLWindow::wnd_proc(
    HWND hwnd,
    UINT msg,
    WPARAM wparam,
    LPARAM lparam)
  {
    return DefWindowProc(hwnd, msg, wparam, lparam);
  }
}
