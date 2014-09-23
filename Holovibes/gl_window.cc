#include "gl_window.hh"

#include <cassert>

namespace gui
{
  bool GLWindow::class_registered_ = false;

  bool GLWindow::wnd_class_register()
  {
    /* Init must be called only if Window Class
    ** has not been registered.
    */
    assert(!class_registered_ && "WndClass has already been registered.");

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

    class_registered_ = RegisterClass(&wc);
    return class_registered_;
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

    assert(hwnd_ && "Failed to initialize window");

    /* Get the device context. */
    hdc_ = GetDC(hwnd_);

    return hwnd_;
  }

  void GLWindow::wnd_show()
  {
    assert(hwnd_ && "Window is not initialized");
    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
  }

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

  void GLWindow::gl_init()
  {
    PIXELFORMATDESCRIPTOR pfd = gl_get_pfd();
    int pixel_format = ChoosePixelFormat(hdc_, &pfd);
    assert(pixel_format && "Unable to find a suitable pixel format");
    bool status = SetPixelFormat(hdc_, pixel_format, &pfd);
    assert(status && "Cannot set the pixel format");

    hrc_ = wglCreateContext(hdc_);
    assert(hrc_ && "Unable to create GL context");

    status = wglMakeCurrent(hdc_, hrc_);
    assert(status && "Unable to make current GL context");
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
