#include "stdafx.h"

#include "option_parser.hh"
#include "holovibes.hh"

int main(const int argc, const char* argv[])
{
  holovibes::OptionParser opts_parser(argc, argv);
  opts_parser.init_parser();
  opts_parser.proceed();
  const holovibes::s_options& opts = opts_parser.get_opt();

  holovibes::Holovibes h(holovibes::Holovibes::XIQ, opts.buffsize);

  if (opts.display_images)
  {
    h.init_camera();
    h.init_display(opts.width_win, opts.height_win);
    while (true)
    {
      h.update_display();
    }
  }
}
