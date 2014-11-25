#include "gl_window.hh"

#include <stdexcept>

namespace holovibes
{
  bool GLWindow::running_ = false;

  GLWindow::GLWindow(
    const char* title,
    int width,
    int height,
    const camera::FrameDescriptor& desc)
    : hinstance_(GetModuleHandle(NULL))
    , hwnd_(nullptr)
    , gl_(nullptr)
  {
    wnd_register_class();

    /* Creation of the window. */
    hwnd_ = CreateWindow(
      "OpenGL",
      title,
      /* WS_OVERLAPPEDWINDOW without WS_THICKFRAME */
      WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME,
      CW_USEDEFAULT, CW_USEDEFAULT,
      width, height,
      NULL, NULL,
      hinstance_, NULL);

    if (!hwnd_)
      throw std::runtime_error("[DISPLAY] failed to instanciate an OpenGL window class");

    /* It do not have default constructor. The constructor of GLComponent
     * must be called once the window has been created. So we can not use
     * the initialization list. It do not provide default constructor because
     * of ~GLComponent() destructor that free OpenGL ressources and things
     * become messy without proper initialization. */
    gl_ = new GLComponent(hwnd_, desc, width, height);
    if (!gl_)
      throw std::runtime_error("[DISPLAY] failed to instanciate GLComponent class.");
  }

  GLWindow::~GLWindow()
  {
    delete gl_;
    wnd_unregister_class();
    /* Make sure the running flag is off. */
    running_ = false;
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

    if (RegisterClass(&wc) == 0)
      throw std::runtime_error("[DISPLAY] unable to register window class");
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
    case WM_CREATE:
      running_ = true;
      break;
    case WM_CLOSE:
      running_ = false;
      DestroyWindow(hwnd);
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      break;
    default:
      return DefWindowProc(hwnd, msg, wparam, lparam);
    }
    return DefWindowProc(hwnd, msg, wparam, lparam);
  }

  void GLWindow::wnd_msgs_handler()
  {
    MSG msg;

    if (PeekMessage(&msg, hwnd_, 0, 0, PM_REMOVE))
    {
      DispatchMessage(&msg);
    }
  }

  void GLWindow::wnd_show()
  {
    if (!hwnd_)
      throw std::runtime_error("[DISPLAY] no window instanciated");

    ShowWindow(hwnd_, SW_SHOW);
    UpdateWindow(hwnd_);
    SetForegroundWindow(hwnd_);
    SetFocus(hwnd_);
  }
}
