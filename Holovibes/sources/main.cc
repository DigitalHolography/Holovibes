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
	 /* float A[] = { 1, 3, 2, 4 };
	float B[] = { 0, 0, 2, 0 };
	float C[] = { 0, 0, 0, 0 };

	float* A_dev;
	cudaMalloc(&A_dev, 4 * sizeof(float));
	cudaMemcpy(A_dev, A, 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* B_dev;
	cudaMalloc(&B_dev, 4 * sizeof(float));
	cudaMemcpy(B_dev, B, 4 * sizeof(float), cudaMemcpyHostToDevice);
	float* C_dev;
	cudaMalloc(&C_dev, 4 * sizeof(float));
	cudaMemcpy(C_dev, C, 4 * sizeof(float), cudaMemcpyHostToDevice);

	float alpha = 1;
	float* alpha_dev;
	cudaMalloc(&alpha_dev, sizeof(float));
	cudaMemcpy(alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice);
	float beta = 1;
	float* beta_dev;
	cudaMalloc(&beta_dev, sizeof(float));
	cudaMemcpy(beta_dev, &beta, sizeof(float), cudaMemcpyHostToDevice);

	cublasSgemm_v2(cuda_tools::CublasHandle::instance(), CUBLAS_OP_T, CUBLAS_OP_T, 2, 2, 2, &alpha, A_dev, 2, B_dev, 2, &beta, C_dev, 2);
	
	cudaDeviceSynchronize();
	cudaStreamSynchronize(0);

	cudaMemcpy(A, A_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(B, B_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(C, C_dev, 4 * sizeof(float), cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < 4; ++i)
	{
		std::cout << A[i] << " ";
	}
	std::cout << "\n";
	for (unsigned i = 0; i < 4; ++i)
	{
		std::cout << B[i] << " ";
	}
	std::cout << "\n";
	for (unsigned i = 0; i < 4; ++i)
	{
		std::cout << C[i] << " ";
	}
	std::cout << "\n"; */

	// return 0;

	// Custom Qt message handler
	qInstallMessageHandler(qt_output_message_handler);

	holovibes::OptionsDescriptor opts;

	holovibes::OptionsParser opts_parser(opts);
	opts_parser.parse(argc, argv);

	holovibes::Holovibes h;
	h.set_compute_desc(opts.compute_desc);
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
		h.get_compute_desc().register_observer(w);

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