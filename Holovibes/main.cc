#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"

int main(int argc, const char* argv[])
{
  holovibes::OptionsDescriptor opts;
  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  holovibes::Holovibes h(opts.camera);

  h.init_capture(opts.queue_size);
  if (opts.is_gl_window_enabled)
    h.init_display(opts.gl_window_width, opts.gl_window_height);
  if (opts.is_recorder_enabled)
    h.init_recorder(opts.recorder_filepath, opts.recorder_n_img);
  h.init_compute();

  getchar();

  h.dispose_compute();
  h.dispose_display();
  h.dispose_recorder();
  h.dispose_capture();
  return 0;
}
