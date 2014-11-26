#include <Windows.h>

#include "options_parser.hh"
#include "holovibes.hh"

#undef min
#include <QtWidgets>

#include "main_window.hh"

int main(int argc, char* argv[])
{
  holovibes::OptionsDescriptor opts;

  holovibes::OptionsParser opts_parser(opts);
  opts_parser.parse(argc, argv);

  holovibes::Holovibes h;
  h.set_compute_desc(opts.compute_desc);

  if (opts.is_gui_enabled)
  {
    /* --- GUI mode --- */
#ifndef _DEBUG
    /* Hides the console window. */
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif /* !_DEBUG */
    
    QApplication a(argc, argv);
    gui::MainWindow w(h);
    w.show();
    h.get_compute_desc().register_observer(w);

    int status = a.exec();

#ifndef _DEBUG
    ShowWindow(GetConsoleWindow(), SW_SHOW);
#endif /* !_DEBUG */

    return status;
  }
  else
  {
    /* --- CLI mode --- */
    try
    {
      h.init_capture(opts.camera, opts.queue_size);

      if (opts.is_compute_enabled)
        h.init_compute();
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
}
