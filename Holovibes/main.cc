#include "stdafx.h"
#include "main.hh"



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
  if (op)
  {
    op->init_parser();
    op->proceed();
  }
  return op;
}

void manage_parsed_info(s_options opt)
{
  Camera *cam;
  if (opt.display_images)
  {
    cam = create_cam(opt);
    GLWindow *win = launch_display(opt);
    draw_in_win(win, cam, opt);
    free_holo(cam, win);
  }
  else if (opt.record)
  {
    cam = create_cam(opt);
    queue::Queue *q = new queue::Queue(opt.width * opt.height * opt.bitdepth, opt.buffsize);
    Recorder *rec = new Recorder(q, opt.record_path, opt.set_size);
    recording(rec, opt, cam, q);
    free_holo(q, rec, cam);
  }
}

void recording(Recorder *rec, s_options opt, Camera *cam, queue::Queue *q)
{
  if (rec && cam && q)
  {
    for (int i = 0; i < opt.nbimages; i++)
    {
      q->enqueue(cam->get_frame());
      rec->record();
    }
  }
}

Camera *create_cam(s_options opt)
{
  Camera *cam = NULL;
  if (opt.cam.compare("pike") == 0)
    cam = new PikeCamera();
  else if (opt.cam.compare("xiq") == 0)
    cam = new XiqCamera();
  else if (opt.cam.compare("ids") == 0)
    int i = 0; // FIX ME
  else
    std::cout << "cam does not exist" << std::endl;
  if (cam)
  {
    cam->init_camera();
    cam->start_acquisition();
  }
  return cam;
}

void free_holo(Camera *cam, GLWindow *win)
{
  if (cam)
    kill_cam(cam);
  if (win)
  {
    win->gl_disable();
    delete win;
  }
}

void free_holo(queue::Queue *q, Recorder *rec, Camera *cam)
{
  if (cam)
    kill_cam(cam);
  if (q)
    delete q;
  if (rec)
    delete rec;
}
void kill_cam(Camera *cam)
{
  if (cam)
  {
    cam->stop_acquisition();
    cam->shutdown_camera();
    delete cam;
  }
}

void draw_in_win(GLWindow *win, Camera *cam, s_options opt)
{
  if (cam && win)
  {
    while (true)
      win->gl_draw(cam->get_frame(), opt.width, opt.height);
  }
}

GLWindow *launch_display(s_options opt)
{
  GLWindow *win = new GLWindow();
  if (win)
  {
    win->wnd_register_class();
    win->wnd_init("Holovibes", opt.width_win, opt.height_win);
    win->gl_init();
    win->gl_enable(opt.width_win, opt.height_win);
    win->wnd_show();
  }
  return win;
}