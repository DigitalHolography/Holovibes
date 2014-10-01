#include "stdafx.h"
#include "main.hh"

using namespace holovibes;

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
    draw_in_win(launch_display(opt));
  }

}



void draw_in_win(GLWindow *win)
{

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