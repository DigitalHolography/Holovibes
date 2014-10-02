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
  manage_parsed_info(op->get_opt());
  delete op;
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
  Camera *cam = create_cam();
  if (opt.display_images)
  {
   GLWindow *win = launch_display(opt);
   draw_in_win(win, cam,opt );
   free_holo(cam, win);
  }
  else
  {
    queue::Queue *q = new queue::Queue(opt.width * opt.height * opt.bitdepth, opt.buffsize );
    Recorder *rec = new Recorder(q, opt.record_path, opt.set_size);
    recording(rec, opt, cam, q);
    free_holo(q, rec, cam);
  }
}

void recording(Recorder *rec, s_options opt, Camera *cam, queue::Queue *q)
{
  for (int i = 0; i < opt.nbimages; i++)
  {
    q->enqueue(cam->get_frame());
    rec->record();
  }
}

Camera *create_cam()
{
  Camera *cam = new PikeCamera();
  cam->init_camera();
  cam->start_acquisition();
  return cam;
}

void free_holo(Camera *cam, GLWindow *win)
{
  kill_cam(cam);
  win->gl_disable();
  delete win;
}

void free_holo(queue::Queue *q, Recorder *rec, Camera *cam)
{
  kill_cam(cam);
  delete q;
  delete rec;
}
void kill_cam(Camera *cam)
{
  cam->stop_acquisition();
  cam->shutdown_camera();
  delete cam;
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
  win->gl_init();
  win->gl_enable(opt.width_win, opt.height_win);
  win->wnd_show();
  return win;
}