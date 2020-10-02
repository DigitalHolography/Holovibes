/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \mainpage Holovibes

    Documentation for developpers. \n
*/

#include <QApplication>
#include <QLocale>
#include <QPixmap>
#include <QSplashScreen>

#include "options_parser.hh"
#include "MainWindow.hh"
#include "logger.hh"

#include <cublas_v2.h>
#include "cublas_handle.hh"
#include "cusolver_handle.hh"

using camera::Endianness;

void qt_output_message_handler(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
	const std::string& str = msg.toStdString();

	// Do not print the QtChart warning message when
	// adding new points to the chart it slows things down a lot
	if (str.find("is NaN or Inf") != str.npos)
		return;

	std::cout << str << "\n";
}

int main(int argc, char* argv[])
{
	// Custom Qt message handler
	qInstallMessageHandler(qt_output_message_handler);

	holovibes::OptionsDescriptor opts;

	holovibes::OptionsParser opts_parser(opts);
	opts_parser.parse(argc, argv);

	holovibes::Holovibes h;
	h.set_cd(opts.cd);
	if (opts.is_gui_enabled)
	{
		/* --- GUI mode --- */
		QLocale::setDefault(QLocale("en_US"));
		QApplication a(argc, argv);
		QSplashScreen splash(QPixmap("holovibes_logo.png"));
		splash.show();

		#ifndef _DEBUG
		/* Hides the console window. */
		//ShowWindow(GetConsoleWindow(), SW_HIDE);
		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
		#endif /* !_DEBUG */

		/* Hide the possibility to close the console while using Holovibes */
		HWND hwnd = GetConsoleWindow();
		HMENU hmenu = GetSystemMenu(hwnd, FALSE);
		EnableMenuItem(hmenu, SC_CLOSE, MF_GRAYED);

		holovibes::gui::MainWindow w(h);
		w.show();
		splash.finish(&w);
		h.get_cd().register_observer(w);

		// Resizing horizontally the window before starting
		w.layout_toggled();

		return a.exec();
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
					opts.file_image_depth >> 3,
					(opts.file_is_big_endian ?
					Endianness::BigEndian : Endianness::LittleEndian)
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