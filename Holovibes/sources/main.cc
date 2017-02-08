#include <Windows.h>

#include "options_parser.hh"
#include "holovibes.hh"

#undef min
#include <QtWidgets>
#include <thread>
#include <chrono>
#include "main_window.hh"
#include "config.hh"
#include "camera_exception.hh"
#include "options_descriptor.hh"
#include "gui_tool.hh"

namespace gui
{
	class MainWindowAccessor;
}

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
		QPixmap pixmap("holovibes_logo.png");
		QSplashScreen splash(pixmap);
		splash.show();
#ifndef _DEBUG
		/* Hides the console window. */
		ShowWindow(GetConsoleWindow(), SW_HIDE);
#endif /* !_DEBUG */
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		/* ----------------------- */
		/*QSurfaceFormat format;
		format.setVersion(4, 0);
		format.setProfile(QSurfaceFormat::CoreProfile);
		QSurfaceFormat::setDefaultFormat(format);*/
		/* ----------------------- */
		gui::MainWindow w(h);
		gui::GuiTool gt(h, &w);
		w.show();
		splash.finish(&w);
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
				camera::FrameDescriptor fd = {
					static_cast<unsigned short>(opts.file_image_width),
					static_cast<unsigned short>(opts.file_image_height),
					static_cast<float>(opts.file_image_depth >> 3),
					static_cast<float>(global::global_config.import_pixel_size),
					(opts.file_is_big_endian ?
						camera::endianness::BIG_ENDIAN : camera::endianness::LITTLE_ENDIAN)
				};
				h.init_import_mode(
					opts.file_src,
					fd,
					false,
					opts.fps,
					opts.spanStart,
					opts.spanEnd,
					global::global_config.input_queue_max_size,
					h);
			}

			if (opts.is_compute_enabled)
				h.init_compute(holovibes::ThreadCompute::PipeType::PIPE);
			h.recorder(opts.recorder_filepath, opts.recorder_n_img);
			h.dispose_compute();
			h.dispose_capture();
		}
		catch (camera::CameraException& e)
		{
			std::cerr << "[CAMERA] " << e.what() << '\n';
			return 1;
		}
		catch (std::exception& e)
		{
			std::cerr << e.what() << '\n';
			return 1;
		}
		return 0;
	}
}