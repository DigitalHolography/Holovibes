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

/*! \file
 *
 * Core class to use HoloVibe  */
#pragma once

#include "thread_compute.hh"
#include "thread_capture.hh"
#include "thread_reader.hh"
#include "recorder.hh"
#include "compute_descriptor.hh"
#include "concurrent_deque.hh"
#include "chart_point.hh"
#include "icamera.hh"

/*! \brief Contains all function and structure needed to computes data */
namespace holovibes
{
	namespace gui
	{
		class MainWindow;
	}
	class Queue;

	/*! \brief Core class to use HoloVibes
	 *
	 * This class does not depends on the user interface (classes under the
	 * holovibes namespace can be seen as a library).
	 *
	 * It contains high-level ressources (Pipe, Camera, Recorder ...). These
	 * ressources are shared between threads and should be allocated in threads
	 * themselves. */
	class Holovibes
	{
	public:
		/*! \brief Available camera models
		 *
		 * Available Cameras are defined here, it is easier for user to select the
		 * Camera he wish to use, instead of loading the corresponding DLL himself.
		 *
		 * The non hardcoded-way would be to search for DLL and build a list of
		 * available cameras. */

		/*! \brief Construct the holovibes object. */
		Holovibes();

		/*! \brief Destroy the holovibes object. */
		~Holovibes();

		/*! \brief Open the camera and launch the ThreadCapture
		 *
		 * Launch the capture thread to continuously acquire frames in input
		 * buffer. */
		void Holovibes::init_capture(const CameraKind c);

		/*! \brief Request the capture thread to stop - Free ressources. */
		void dispose_capture();

		/*! \brief Check if camera is initialized. */
		bool is_camera_initialized();

		/*! \brief Returns the camera name. */
		const char* get_camera_name();

		/*! \{ \name Queue getters
		 *
		 * Used to record frames */
		std::unique_ptr<Queue>& get_gpu_input_queue()
		{
			return gpu_input_queue_;
		}

		/*! Used to display frames */
		std::unique_ptr<Queue>& get_gpu_output_queue()
		{
			return gpu_output_queue_;
		}

		std::unique_ptr<Queue>& get_current_window_output_queue();
		/*! \} */

		/*! \brief Launch the recorder
		 *
		 * \param filepath File path to record frames
		 * \param rec_n_images Number of frames to record
		 *
		 * The Recorder is used only for CLI purpose, a thread is
		 * available for GUI because it use QThread way (use slots and is
		 * stoppable).
		 * Recorder input queue depends on the mode :
		 *
		 * - raw: use gpu_input_queue_ queue
		 * - hologram: use gpu_output_queue_ queue. */
		void recorder(const std::string& filepath, const unsigned int rec_n_images);

		/*! \brief Launch the ThreadCompute
		 * \see ThreadCompute
		 * \see Pipe
		 *
		 * The pipe is allocated and his allocation can take some times so that
		 * the method contains a lock to avoid conflicts between threads that would
		 * use the Pipe before it finished the initialization. */
		void init_compute(const ThreadCompute::PipeType pipetype = ThreadCompute::PipeType::PIPE, const unsigned int& depth = 2);


		/*! \brief Request the computing thread to stop */
		void dispose_compute();

		/*! \brief Fetch all the necessary information before importing a file. */
		void init_import_mode(std::string &file_src,
			camera::FrameDescriptor fd,
			bool loop,
			unsigned int fps,
			unsigned int spanStart,
			unsigned int spanEnd,
			bool load_file_in_gpu,
			unsigned int q_max_size_,
			Holovibes& holovibes,
			QProgressBar *reader_progress_bar = nullptr,
			gui::MainWindow *main_window = nullptr);

		/*! \{ \name Getters/Setters */
		std::shared_ptr<ICompute> get_pipe()
		{
			if (tcompute_ != nullptr)
				return tcompute_->get_pipe();
			throw std::runtime_error("cannot get pipe, no compute thread");
		}

		/*! \return Common ComputeDescriptor */
		ComputeDescriptor& get_cd()
		{
			return cd_;
		}

		/*! \brief Clear values related to convolution matrix */
		void clear_convolution_matrix();

		/*! \brief Set ComputeDescriptor options
		 *
		 * \param cd ComputeDescriptor to load
		 *
		 * Used when options are loaded from an INI file. */
		void set_cd(const ComputeDescriptor& cd)
		{
			cd_ = cd;
		}

		/*! \return Corresponding Camera INI file path */
		const char* get_camera_ini_path() const
		{
			return camera_->get_ini_path();
		}

		const camera::FrameDescriptor& get_capture_fd();

		/* \brief Get zb = N d^2 / lambda
		  Is updated everytime the camera changes or lamdba changes
		  */
		const float get_boundary();

		/* \brief Getter onto launch_path
		*/
		const std::string get_launch_path();

		/* \brief Return capture thread*/
		std::unique_ptr<IThreadInput>& get_tcapture()
		{
			return tcapture_;
		}

	private:
		/* Use shared pointers to ensure each ressources will freed. */
		/*! \brief ICamera use to acquire image */
		std::shared_ptr<camera::ICamera> camera_;
		bool camera_initialized_;
		/*! \brief IThread which acquiring continuously frames */
		std::unique_ptr<IThreadInput> tcapture_;
		/*! \brief Thread which compute continuously frames */
		std::unique_ptr<ThreadCompute> tcompute_;

		/*! \{ \name Frames queue (GPU) */
		std::unique_ptr<Queue> gpu_input_queue_;
		std::unique_ptr<Queue> gpu_output_queue_;
		/*! \} */

		/*! \brief Common compute descriptor shared between CLI/GUI and the
		 * Pipe. */
		ComputeDescriptor cd_;

		/* \brief Store the path of holovibes when it is launched.
		   so that holovibes.ini is saved at the right place. The problem
		   is that QT's functions actually change the current directory so
		   saving holovibes.ini in "$PWD" isn't working*/
		std::string launch_path;

		mutable std::mutex mutex_;
	};
}
