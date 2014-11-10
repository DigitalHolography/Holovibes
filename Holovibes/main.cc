#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"
#include "camera_exception.hh"

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"

int main(int argc, const char* argv[])
{
  holovibes::OptionsDescriptor opts;
  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  try
  {
    holovibes::Holovibes h(opts.camera);

    h.init_capture(opts.queue_size);

    if (opts.is_1fft_enabled || opts.is_2fft_enabled)
      h.init_compute(opts.compute_desc);
    if (opts.is_gl_window_enabled)
      h.init_display(opts.gl_window_width, opts.gl_window_height);
    if (opts.is_recorder_enabled)
      h.init_recorder(opts.recorder_filepath, opts.recorder_n_img);

    std::cout << "Press any key to stop execution..." << std::endl;
    getchar();

    h.dispose_display();
    h.dispose_recorder();
    h.dispose_compute();
    h.dispose_capture();
  }
  catch (camera::CameraException& e)
  {
    std::cerr << "[CAMERA] " << e.get_name() << " " << e.what() << std::endl;
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }
  return 0;
}
