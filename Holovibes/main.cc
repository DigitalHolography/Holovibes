#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"

int main(int argc, const char* argv[])
{
  holovibes::OptionsDescriptor opts;
  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  holovibes::Holovibes h(opts.camera);

  if (opts.is_gl_window_enabled)
  {
    h.init_capture(opts.queue_size);
    h.init_display(opts.gl_window_width, opts.gl_window_height);
    getchar();
    h.dispose_display();
    h.dispose_capture();
  }

  return 0;
}
