#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"
#include "camera.hh"
#include "camera_exception.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"

#undef min
#include <QtWidgets>

#include "main_window.hh"
#include "gui_gl_window.hh"
#include "gui_gl_widget.hh"

#include "compute_descriptor.hh"
#include "pipeline.hh"

#include <thread>

int main(int argc, char* argv[])
{
  // Holovibes object
  holovibes::ComputeDescriptor cd;
  cd.algorithm = holovibes::ComputeDescriptor::FFT1;
  cd.shift_corners_enabled = false;
  cd.pindex = 0;
  cd.nsamples = 2;
  cd.lambda = 536e-9f;
  cd.zdistance = 1.36f;

  holovibes::Holovibes h(holovibes::Holovibes::camera_type::IDS);
  h.init_capture(20);
  holovibes::Pipeline* pipeline = h.init_compute(cd);

  if (!pipeline)
  {
    std::cout << "pipeline null" << std::endl;
    getchar();
  }

  // GUI
  QApplication a(argc, argv);
  gui::MainWindow w(pipeline);
  w.show();
  gui::GuiGLWindow glw(&w);

  unsigned int gl_width = 512;
  unsigned int gl_height = 512;

  gui::GLWidget glwi(&glw, h.get_output_queue(), gl_width, gl_height);
  glwi.setObjectName("GL");
  glwi.resize(glwi.sizeHint());
  glwi.show();
  glw.show();

  return a.exec();
}

/*int main(int argc, const char* argv[])
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
}*/
