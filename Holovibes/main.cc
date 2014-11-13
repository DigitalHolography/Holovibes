#include "stdafx.h"

#include "options_parser.hh"
#include "holovibes.hh"
#include "camera_exception.hh"

#include "camera.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"

#undef min
#include <QtWidgets>

#include "main_window.hh"
#include "gui_gl_window.hh"
#include "gui_gl_widget.hh"

#include "camera.hh"
#include "camera_ids.hh"
#include "camera_pixelfly.hh"
#include <thread>

int main(int argc, char* argv[])
{
  // Holovibes object
  holovibes::Holovibes h(holovibes::Holovibes::camera_type::PIXELFLY);
  h.init_capture(20);

  // GUI
  QApplication a(argc, argv);
  gui::MainWindow w;
  w.show();
  gui::GuiGLWindow glw(&w);

  unsigned int gl_width = 512;
  unsigned int gl_height = 512;

  gui::GLWidget glwi(&glw, h.get_capture_queue(), gl_width, gl_height);
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

    h.init_compute(opts.p, opts.nbimages, opts.lambda, opts.distance);

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
