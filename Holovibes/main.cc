#include "stdafx.h"
#include <iostream>
#include <Windows.h>
#include "gl_window.hh"

#include "ids_camera.hh"
#define WIDTH 512
#define HEIGHT 512

int main()
{
  holovibes::GLWindow window;

  window.wnd_register_class();
  window.wnd_init("IDS", WIDTH, HEIGHT);
  window.gl_init();
  window.gl_enable(WIDTH, HEIGHT);
  window.wnd_show();

  camera::IDSCamera cam;
  cam.init_camera();
  cam.start_acquisition();

  while (true)
    window.gl_draw(cam.get_frame(),
    cam.get_frame_descriptor().width,
    cam.get_frame_descriptor().height);

  cam.stop_acquisition();
  cam.shutdown_camera();

  window.gl_free();
  window.gl_disable();
  window.wnd_unregister_class();

  getchar();
  return 0;
}
