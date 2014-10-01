#ifndef MAIN_HH
# define MAIN_HH
#include <iostream>
#include "option_parser.hh"
#include "camera.hh"
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <Windows.h>
#include "gl_window.hh"

using namespace holovibes;
using namespace camera;

OptionParser *gen_opt_parser(const int argc, const char *argv[]);
void manage_parsed_info(s_options opt);
void draw_in_win(GLWindow *win);
GLWindow *launch_display(s_options opt);
void draw_in_win(GLWindow *win, Camera *cam, s_options opt);

#endif