#include "stdafx.h"
#include "main.hh"
#include <iostream>
#include <fstream>
#include <cstdio>
#include "pike_camera.hh"
#include "queue.hh"


using namespace holovibes;
using namespace camera;

int main(const int argc, const char *argv[])
{
  OptionParser *op = gen_opt_parser(argc, argv);
  
  std::cout << "Hello World!" << std::endl;

  return 0;
}

OptionParser *gen_opt_parser(const int argc, const char *argv[])
{
  OptionParser *op = new OptionParser(argc, argv);
  op->init_parser();
  op->proceed();
  return op;
}

void manage_parsed_info(s_options opt)
{
  if (opt.display_images)
  {
    //draw_in_win(launch_display(opt));
  }
  else
  {
  }

}

Camera *create_cam()
{
  return (new PikeCamera());
}

void draw_in_win(GLWindow *win, Camera *cam, s_options opt)
{
  for (int i = 0; i < opt.nbimages; i++)
    win->gl_draw(cam->get_frame(), opt.width, opt.height);
}

GLWindow *launch_display(s_options opt)
{

  GLWindow *win = new GLWindow();
  win->wnd_register_class();
  win->wnd_init("Holovibes", opt.width_win, opt.height_win);
  win->wnd_show();
  win->gl_init();
  win->gl_enable(opt.width_win, opt.height_win);
  // draw miss
  return win;
}