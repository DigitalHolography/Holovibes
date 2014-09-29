#include "stdafx.h"
#include <iostream>
#include "camera.hh"
#include "xiq_camera.hh"
#include "gl_window.hh"

using namespace camera;
using namespace holovibes;

#define WIDTH 512
#define HEIGHT 512

int main()
{
  Camera* c = new XiqCamera();

  c->init_camera();

  GLWindow w;
  w.wnd_register_class();
  w.wnd_init("Test", WIDTH, HEIGHT);
  w.gl_init();
  w.gl_enable(WIDTH, HEIGHT);
  w.wnd_show();

  c->start_acquisition();

  while (true)
    w.gl_draw(c->get_frame(), 2048, 2048);

  c->stop_acquisition();
  w.gl_disable();
  w.gl_free();

  c->shutdown_camera();

  return 0;
}