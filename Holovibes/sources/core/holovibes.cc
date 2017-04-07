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

#include <exception>
#include <cassert>
#include <memory>
#include <boost/filesystem.hpp>

#include "holovibes.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "icamera.hh"
#include "config.hh"
#include "camera_dll.hh"
#include "tools.hh"

namespace holovibes
{
	Holovibes::Holovibes()
		: camera_()
		, camera_initialized_(false)
		, tcapture_()
		, tcompute_()
		, input_()
		, output_()
		, compute_desc_()
		, average_queue_()
		, launch_path(boost::filesystem::current_path().generic_string())
	{
	}

	Holovibes::~Holovibes()
	{
	}

	void Holovibes::init_capture(const enum camera_type c)
	{
		camera_initialized_ = false;
		try
		{
			if (c == ADIMEC)
				camera_ = camera::CameraDLL::load_camera("CameraAdimec.dll");
			else if (c == EDGE)
				camera_ = camera::CameraDLL::load_camera("CameraPCOEdge.dll");
			else if (c == IDS)
				camera_ = camera::CameraDLL::load_camera("CameraIds.dll");
			else if (c == IXON)
				camera_ = camera::CameraDLL::load_camera("CameraIxon.dll");
			else if (c == PIKE)
				camera_ = camera::CameraDLL::load_camera("CameraPike.dll");
			else if (c == PIXELFLY)
				camera_ = camera::CameraDLL::load_camera("CameraPCOPixelfly.dll");
			else if (c == XIQ)
				camera_ = camera::CameraDLL::load_camera("CameraXiq.dll");
			else
				assert(!"Impossible case");

			std::cout << "(Holovibes) Prepared to initialize camera." << std::endl;
			camera_->init_camera();
			std::cout << "(Holovibes) Prepared to reset queues." << std::endl;
			input_.reset(new Queue(camera_->get_frame_descriptor(), global::global_config.input_queue_max_size, "InputQueue"));
			std::cout << "(Holovibes) Prepared to start initialization." << std::endl;
			camera_->start_acquisition();
			tcapture_.reset(new ThreadCapture(*camera_, *input_));

			std::cout << "[CAPTURE] capture thread started" << std::endl;
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

		std::cout << "[CAPTURE] capture thread stopped" << std::endl;
	}

	void Holovibes::recorder(
		const std::string& filepath,
		const unsigned int rec_n_images)
	{

		assert(camera_initialized_ && "camera not initialized");
		assert(tcapture_ && "capture thread not initialized");

		Recorder* recorder = new Recorder(
			*((tcompute_) ? output_ : input_),
			filepath);

		std::cout << "[RECORDER] recorder Start" << std::endl;
		recorder->record(rec_n_images);
		delete recorder;
		std::cout << "[RECORDER] recorder Stop" << std::endl;
	}

	void Holovibes::init_compute(const ThreadCompute::PipeType pipetype, const unsigned int& depth)
	{
		assert(camera_initialized_ && "camera not initialized");
		assert(tcapture_ && "capture thread not initialized");
		assert(input_ && "input queue not initialized");

		camera::FrameDescriptor output_fd = input_->get_frame_desc();
		/* depth is 2 by default execpt when we want dynamic complex dislay*/
		output_fd.depth = depth;
		output_.reset(new Queue(
			output_fd, global::global_config.output_queue_max_size, "OutputQueue"));

		tcompute_.reset(new ThreadCompute(compute_desc_, *input_, *output_, pipetype));
		std::cout << "[CUDA] compute thread started" << std::endl;

		// A wait_for is necessary here in order for the pipe to finish
		// its allocations before getting it.
		std::unique_lock<std::mutex> lock(mutex_);

		std::cout << "Pipe is initializing ";
		while (tcompute_->get_memory_cv().wait_for(
			lock, std::chrono::milliseconds(100)) == std::cv_status::timeout)
		{
			std::cout << ".";
		}
		std::cout << std::endl;
	}

	void Holovibes::dispose_compute()
	{
		tcompute_.reset(nullptr);
		output_.reset(nullptr);
	}

	void Holovibes::reset_convolution_matrix()
	{
		compute_desc_.convo_matrix_width.exchange(0);
		compute_desc_.convo_matrix_height.exchange(0);
		compute_desc_.convo_matrix_z.exchange(0);
		compute_desc_.convo_matrix.clear();
	}

	const camera::FrameDescriptor& Holovibes::get_cam_frame_desc()
	{
		return (tcapture_.get()->get_frame_descriptor());
	}

	const float Holovibes::get_boundary()
	{
		if (tcapture_)
		{
			FrameDescriptor fd = get_cam_frame_desc();
			const float n = static_cast<float>(fd.height);
			const float d = fd.pixel_size * /*fd.binning **/ 0.000001f;
			return ((n * d * d) / compute_desc_.lambda.load());
		}
		return (0.f);
	}

	void Holovibes::init_import_mode(std::string &file_src,
		camera::FrameDescriptor frame_desc,
		bool loop,
		unsigned int fps,
		unsigned int spanStart,
		unsigned int spanEnd,
		unsigned int q_max_size_,
		Holovibes& holovibes)
	{
		camera_initialized_ = false;

		try
		{
			unsigned short	size = nearest_window_size(frame_desc);
			camera::FrameDescriptor real_frame_desc = frame_desc;
			real_frame_desc.width = size;
			real_frame_desc.height = size;
			input_.reset(new Queue(real_frame_desc, q_max_size_, "InputQueue"));
			tcapture_.reset(
				new ThreadReader(file_src
					, real_frame_desc
					, frame_desc
					, loop
					, fps
					, spanStart
					, spanEnd
					, *input_
					, compute_desc_.is_cine_file.load()
					, holovibes));
			std::cout << "[CAPTURE] reader thread started" << std::endl;
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
