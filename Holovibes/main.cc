#include <Windows.h>

#include "options_parser.hh"
#include "holovibes.hh"

#undef min
#include <QtWidgets>

#include "main_window.hh"
#include "config.hh"

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
    QApplication a(argc, argv);
#ifndef _DEBUG
    /* Hides the console window. */
    ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif /* !_DEBUG */

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
      if (!opts.is_import_mode_enabled)
        h.init_capture(opts.camera);
      else
      {
        h.init_import_mode(
          opts.file_src,
          holovibes::ThreadReader::FrameDescriptor({
          opts.file_image_width,
          opts.file_image_height,
          opts.file_image_depth / 8,
          0.0f,
          (opts.file_is_big_endian ? camera::endianness::BIG_ENDIAN : camera::endianness::LITTLE_ENDIAN),
        }),
        false,
        opts.fps,
        opts.spanStart,
        opts.spanEnd,
        Global::global_config.input_queue_max_size);
      }

      if (opts.is_compute_enabled)
        h.init_compute(holovibes::ThreadCompute::PipeType::PIPE);
      h.recorder(opts.recorder_filepath, opts.recorder_n_img);
      h.dispose_compute();
      h.dispose_capture();
    }
    catch (camera::CameraException& e)
    {
      std::cerr << "[CAMERA] " << e.what() << std::endl;
      return 1;
    }
    catch (std::exception& e)
    {
      std::cerr << e.what() << std::endl;
      return 1;
    }
    return 0;
  }
}