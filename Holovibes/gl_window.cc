#include "gl_window.hh"
#include <Windows.h>
#include <gl\GL.h>

namespace gui
{
  GLWindow* GLWindow::instance_ = nullptr;

  bool GLWindow::init(
    const char* title,
    int width,
    int height)
  {
    /* Window class structure */
    WNDCLASS wc;

    memset(&wc, 0, sizeof (WNDCLASS));
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = (WNDPROC)wnd_proc;
    wc.hInstance = hinstance_;
    wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "OpenGL";

    if (!RegisterClass(&wc))
      return false;

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
      shutdown();
      return false;
    }

    ShowWindow(hwnd_, SW_SHOW);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);

    return true;
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
