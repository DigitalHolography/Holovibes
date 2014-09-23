#include "gl_window.hh"
#include <Windows.h>
#include <gl\GL.h>

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

    memset(&wc, 0, sizeof (WNDCLASS));
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = (WNDPROC)wnd_proc;
    wc.hInstance = hinstance_;
    wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
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

    return hwnd_;
  }

  void GLWindow::wnd_show()
  {
    assert(hwnd_ && "Window is not initialized");
    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
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
