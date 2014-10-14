#include "stdafx.h"
#include "gl_window.hh"
#include "error_handler.hh"

#include <exception>

namespace holovibes
{
  GLWindow::GLWindow(
    const char* title,
    int width,
    int height)
    : hinstance_(GetModuleHandle(NULL))
    , hwnd_(nullptr)
    , gl_(nullptr)
  {
    wnd_register_class();

    /* Creation of the window. */
    hwnd_ = CreateWindow(
      "OpenGL",
      title,
      WS_CLIPSIBLINGS | WS_CLIPCHILDREN,//WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME, // REMOVE WS_THICKFRAME responsible for window resizing.
      CW_USEDEFAULT, CW_USEDEFAULT,
      width, height,
      NULL, NULL,
      hinstance_, NULL);

    if (!hwnd_)
      throw std::exception("failed to instanciate an OpenGL window class");

    gl_ = new GLComponent(hwnd_, width, height);
    if (!gl_)
      throw std::exception("failed to instanciate GLComponent class.");
  }

  GLWindow::~GLWindow()
  {
    delete gl_;
    wnd_unregister_class();
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

  LRESULT GLWindow::wnd_proc(
    HWND hwnd,
    UINT msg,
    WPARAM wparam,
    LPARAM lparam)
  {
    switch (msg)
    {
    case WM_CLOSE:
      DestroyWindow(hwnd);
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      break;
    default:
      return DefWindowProc(hwnd, msg, wparam, lparam);
    }
  }

  void GLWindow::wnd_show()
  {
    if (!hwnd_)
      throw std::exception("no window instanciated");

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
  }
}
