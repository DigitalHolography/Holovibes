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

#include <filesystem>

#include <cassert>

#include "holovibes.hh"
#include "queue.hh"
#include "config.hh"
#include "camera_dll.hh"
#include "tools.hh"
#include "logger.hh"
#include "holo_file.hh"

namespace holovibes
{
	using camera::FrameDescriptor;

	Holovibes::Holovibes()
		: camera_(),
		camera_initialized_(false),
		tcapture_(),
		tcompute_(),
		input_(),
		output_(),
		cd_(),
		average_queue_(),
		launch_path(std::filesystem::current_path().generic_string())
	{
	}

	Holovibes::~Holovibes()
	{
	}

	void Holovibes::init_capture(const CameraKind c)
	{
		camera_initialized_ = false;
		try
		{
			if (c == CameraKind::Adimec)
				camera_ = camera::CameraDLL::load_camera("CameraAdimec.dll");
			else if (c == CameraKind::Edge)
				camera_ = camera::CameraDLL::load_camera("CameraPCOEdge.dll");
			else if (c == CameraKind::IDS)
				camera_ = camera::CameraDLL::load_camera("CameraIds.dll");
			else if (c == CameraKind::Ixon)
				camera_ = camera::CameraDLL::load_camera("CameraIxon.dll");
			else if (c == CameraKind::Hamamatsu)
				camera_ = camera::CameraDLL::load_camera("CameraHamamatsu.dll");
			else if (c == CameraKind::Pike)
				camera_ = camera::CameraDLL::load_camera("CameraPike.dll");
			else if (c == CameraKind::Pixelfly)
				camera_ = camera::CameraDLL::load_camera("CameraPCOPixelfly.dll");
			else if (c == CameraKind::Pixelink)
				camera_ = camera::CameraDLL::load_camera("CameraPixelink.dll");
			else if (c == CameraKind::xiQ)
				camera_ = camera::CameraDLL::load_camera("CameraXiq.dll");
			else if (c == CameraKind::xiB)
				camera_ = camera::CameraDLL::load_camera("CameraXib.dll");
			else if (c == CameraKind::PhotonFocus)
				camera_ = camera::CameraDLL::load_camera("CameraPhotonFocus.dll");
			else
				assert(!"Impossible case");

			LOG_INFO("(Holovibes) Initializing camera...");
			camera_->init_camera();
			cd_.pixel_size = camera_->get_pixel_size();
			LOG_INFO("(Holovibes) Resetting queues...");

			auto camera_fd = camera_->get_fd();
			auto queue_fd = camera_fd;
			SquareInputMode mode = cd_.square_input_mode;
			//unsigned short	size = upper_window_size(fd.width, fd.height);
			if (mode == SquareInputMode::ZERO_PADDED_SQUARE)
			{
				//Set values to the max of the two
				set_max_of_the_two(queue_fd.width, queue_fd.height);
			}
			else if (mode == SquareInputMode::CROPPED_SQUARE)
			{
				//Set values to the min of the two
				set_min_of_the_two(queue_fd.width, queue_fd.height);
			}

			input_.reset(new Queue(queue_fd, global::global_config.input_queue_max_size, "InputQueue", camera_fd.width, camera_fd.height, camera_fd.depth));

			LOG_INFO("(Holovibes) Starting initialization...");
			camera_->start_acquisition();
			tcapture_.reset(new ThreadCapture(*camera_, *input_, mode));
			LOG_INFO("[CAPTURE] Capture thread started");
			camera_initialized_ = true;
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			tcapture_.reset(nullptr);
			input_.reset(nullptr);

			throw;
		}
	}

	void Holovibes::dispose_capture()
	{
		tcapture_.reset(nullptr);
		if (camera_ && camera_initialized_)
		{
			camera_->stop_acquisition();
			camera_->shutdown_camera();
		}

		input_.reset(nullptr);
		camera_.reset();
		camera_initialized_ = false;

		LOG_INFO("[CAPTURE] Capture thread stopped");
	}

	bool Holovibes::is_camera_initialized()
	{
		return camera_.operator bool();
	}

	const char* Holovibes::get_camera_name()
	{
		assert(camera_initialized_ && "Camera not initialized");
		return camera_.get()->get_name();
	}

	std::unique_ptr<Queue>& Holovibes::get_current_window_output_queue()
	{
		if (cd_.current_window == WindowKind::XYview)
			return output_;
		else if (cd_.current_window == WindowKind::XZview)
			return get_pipe()->get_stft_slice_queue(0);
		return get_pipe()->get_stft_slice_queue(1);
	}

	void Holovibes::recorder(const std::string& filepath, const unsigned int rec_n_images)
	{

		assert(camera_initialized_ && "Camera not initialized");
		assert(tcapture_ && "Capture thread not initialized");

		Recorder recorder = Recorder(*((tcompute_) ? output_ : input_),
			filepath);

		LOG_INFO("[RECORDER] Recorder Start");
		recorder.record(rec_n_images, HoloFile::get_json_settings(cd_, get_output_queue()->get_fd()));
		LOG_INFO("[RECORDER] Recorder Stop");
	}

	void Holovibes::init_compute(const ThreadCompute::PipeType pipetype, const unsigned int& depth)
	{
		assert(camera_initialized_ && "Camera not initialized");
		assert(tcapture_ && "Capture thread not initialized");
		assert(input_ && "Input queue not initialized");

		camera::FrameDescriptor output_fd = input_->get_fd();
		/* depth is 2 by default execpt when we want dynamic complex dislay*/
		output_fd.depth = depth;
		try
		{
			output_.reset(new Queue(
				output_fd, global::global_config.output_queue_max_size, "OutputQueue"));
		}
		catch (std::logic_error& e)
		{
			std::cerr << e.what() << std::endl;
			tcapture_.reset(nullptr);
			input_.reset(nullptr);
			return;
		}
		tcompute_.reset(new ThreadCompute(cd_, *input_, *output_, pipetype));
		LOG_INFO("[CUDA] Compute thread started");

		// A wait_for is necessary here in order for the pipe to finish
		// its allocations before getting it.
		std::unique_lock<std::mutex> lock(mutex_);

		LOG_INFO("Pipe is initializing ");
		while (tcompute_->get_memory_cv().wait_for(
			lock, std::chrono::milliseconds(100)) == std::cv_status::timeout)
		{
			std::cout << ".";
		}
		std::cout << std::endl;
		LOG_INFO("Pipe initialized.");
	}

	void Holovibes::dispose_compute()
	{
		tcompute_.reset(nullptr);
		output_.reset(nullptr);
	}

	void Holovibes::clear_convolution_matrix()
	{
		cd_.convo_matrix_width = 0;
		cd_.convo_matrix_height = 0;
		cd_.convo_matrix_z = 0;
		cd_.convo_matrix.clear();
	}

	const camera::FrameDescriptor& Holovibes::get_capture_fd()
	{
		return tcapture_->get_queue_fd();
	}

	const float Holovibes::get_boundary()
	{
		if (tcapture_)
		{
			FrameDescriptor fd = get_capture_fd();
			const float n = static_cast<float>(fd.height);
			const float d = cd_.pixel_size * 0.000001f;
			return (n * d * d) / cd_.lambda;
		}
		return 0.f;
	}

	void Holovibes::init_import_mode(std::string &file_src,
		camera::FrameDescriptor fd,
		bool loop,
		unsigned int fps,
		unsigned int spanStart,
		unsigned int spanEnd,
		bool load_file_in_gpu,
		unsigned int q_max_size_,
		Holovibes& holovibes,
		QProgressBar *reader_progress_bar,
		gui::MainWindow *main_window)
	{
		camera_initialized_ = false;

		try
		{
			SquareInputMode mode = cd_.square_input_mode;
			camera::FrameDescriptor queue_fd = fd;
			if (mode == SquareInputMode::ZERO_PADDED_SQUARE)
			{
				//Set values to the max of the two
				set_max_of_the_two(queue_fd.width, queue_fd.height);
			}
			else if (mode == SquareInputMode::CROPPED_SQUARE)
			{
				//Set values to the min of the two
				set_min_of_the_two(queue_fd.width, queue_fd.height);
			}

			input_.reset(new Queue(queue_fd, q_max_size_, "InputQueue", fd.width, fd.height, fd.depth));
			tcapture_.reset(
				new ThreadReader(file_src,
					fd,
					mode,
					loop,
					fps,
					spanStart,
					spanEnd,
					*input_,
					cd_.file_type,
					load_file_in_gpu,
					reader_progress_bar,
					main_window));
			LOG_INFO("[CAPTURE] reader thread started");
			camera_initialized_ = true;
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
			tcapture_.reset(nullptr);
			input_.reset(nullptr);

			throw;
		}
	}

	const std::string Holovibes::get_launch_path()
	{
		return launch_path;
	}
}
