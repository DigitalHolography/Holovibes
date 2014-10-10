#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"
#include "thread_capture.hh"

int main(int argc, const char* argv[])
{
  holovibes::OptionsDescriptor opts;
  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  holovibes::Holovibes h(opts.camera, opts.queue_size);

  if (opts.is_gl_window_enabled)
  {
    h.init_camera();
    h.init_display(opts.gl_window_width, opts.gl_window_height);

    holovibes::ThreadCapture t(*h.camera_, *h.queue_);

    while (true)
      h.update_display();
  }
  getchar();
  return 0;
}
