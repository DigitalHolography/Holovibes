#include "options_parser.hh"
#include "holovibes.hh"
#include "camera.hh"
#include "camera_exception.hh"
#include "camera_pixelfly.hh"
#include "queue.hh"
#include "camera_ixon.hh"


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
	camera::CameraIxon c = camera::CameraIxon();
	c.init_camera();
	c.start_acquisition();
	Sleep(2000);
	void *imgs = malloc(c.get_frame_descriptor().frame_size() * 100);
	for (int i = 0; i < 100; i++)
	{
		void *frame = c.get_frame();
		memcpy((unsigned char*)imgs + i * c.get_frame_descriptor().frame_size(), frame, c.get_frame_descriptor().frame_size());
	}
	FILE *f;
	fopen_s(&f,"ixon.raw","w+b");
	fwrite(imgs, c.get_frame_descriptor().frame_size(), 100, f);
	fclose(f);
	exit(0);
	
  // Holovibes object
  holovibes::Holovibes h(holovibes::Holovibes::camera_type::IDS);
  h.init_capture(20);

  // GUI
  QApplication a(argc, argv);
  gui::MainWindow w(h);
  w.show();
  h.get_compute_desc().register_observer(w);

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
