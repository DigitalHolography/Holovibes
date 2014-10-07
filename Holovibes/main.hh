#ifndef MAIN_HH
# define MAIN_HH
#include <iostream>
#include "option_parser.hh"
#include "camera.hh"
#include "recorder.hh"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <iterator>
#include <algorithm>
#include <Windows.h>
#include "camera_xiq.hh"
#include "gl_window.hh"
#include "stdafx.h"
#include <fstream>
#include <cstdio>
#include "camera_pike.hh"
#include "queue.hh"
#include "exception_camera.hh"
#include "camera_ids.hh"
#include "camera_pixelfly.hh"

using namespace holovibes;
using namespace camera;

OptionParser *gen_opt_parser(const int argc, const char *argv[]);
void manage_parsed_info(s_options opt);
Camera *create_cam(s_options opt);
void draw_in_win(GLWindow *win);
GLWindow *launch_display(s_options opt);
void draw_in_win(GLWindow *win, Camera *cam, s_options opt);
void free_holo(Camera *cam, GLWindow *win);
void free_holo(queue::Queue *q, Recorder *rec, Camera *cam);
void kill_cam(Camera *cam);
void recording(Recorder *rec, s_options opt, Camera *cam, queue::Queue *q);

#endif