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

#include "compute_descriptor.hh"
#include "icamera.hh"

#include "concurrent_deque.hh"
#include "chart_point.hh"

// Worker & Controller
#include "thread_worker_controller.hh"
#include "file_frame_read_worker.hh"
#include "camera_frame_read_worker.hh"
#include "information_worker.hh"
#include "chart_record_worker.hh"
#include "frame_record_worker.hh"
#include "batch_gpib_worker.hh"
#include "compute_worker.hh"

// Enum
#include "enum_camera_kind.hh"

#include "information_container.hh"

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
		static Holovibes& instance();

		/*! \{ \name Queue getters
		 *
		 * Used to record frames */
		std::shared_ptr<Queue> get_gpu_input_queue();

		/*! Used to display frames */
		std::shared_ptr<Queue> get_gpu_output_queue();

		Queue* get_current_window_output_queue();
		/*! \} */

		/*! \{ \name Getters/Setters */
		std::shared_ptr<ICompute> get_compute_pipe();

		/*! \return Common ComputeDescriptor */
		ComputeDescriptor& get_cd();

		/*! \brief Set ComputeDescriptor options
		 *
		 * \param cd ComputeDescriptor to load
		 *
		 * Used when options are loaded from an INI file. */
		void set_cd(const ComputeDescriptor& cd);

		/*! \return Corresponding Camera INI file path */
		const char* get_camera_ini_path() const;

		/* \brief Get zb = N d^2 / lambda
		  Is updated everytime the camera changes or lamdba changes
		  */
		const float get_boundary();

		InformationContainer& get_info_container();

		/*! \brief Update the compute descriptor for CLI purpose
		 * Must be called before the initialization of the thread compute and recorder
		 */
		void update_cd_for_cli(const unsigned int input_fps);

		void init_input_queue(const camera::FrameDescriptor& fd);

		void start_file_frame_read(const std::string& file_path,
									bool loop,
									unsigned int fps,
									unsigned int first_frame_id,
									unsigned int nb_frames_to_read,
									bool load_file_in_gpu,
									const std::function<void()>& callback = [](){});

		void start_camera_frame_read(CameraKind camera_kind, const std::function<void()>& callback = [](){});

		void stop_frame_read();

		/*! \brief Clear values related to convolution matrix */
		void clear_convolution_matrix();

		void start_frame_record(const std::string& path, std::optional<unsigned int> nb_frames_to_record,
								bool raw_record, const std::function<void()>& callback = [](){});

		void stop_frame_record();

		void start_chart_record(const std::string& path, const unsigned int nb_points_to_record,
			const std::function<void()>& callback = [](){});

		void stop_chart_record();

		void start_batch_gpib(const std::string& batch_input_path,
            				const std::string& output_path,
							unsigned int nb_frames_to_record,
                			bool chart_record,
							bool raw_record_enabled,
							const std::function<void()>& callback = [](){});

		void stop_batch_gpib();

		void start_information_display(bool is_cli, const std::function<void()>& callback = [](){});

		void stop_information_display();

		void start_compute(const std::function<void()>& callback = [](){});

		void stop_compute();

		void stop_all_worker_controller();

	private:
		/*! \brief Construct the holovibes object. */
		Holovibes() = default;

		InformationContainer info_container_;

		worker::ThreadWorkerController<worker::FileFrameReadWorker> file_read_worker_controller_;
		worker::ThreadWorkerController<worker::CameraFrameReadWorker> camera_read_worker_controller_;
		std::shared_ptr<camera::ICamera> active_camera_{ nullptr };

		worker::ThreadWorkerController<worker::FrameRecordWorker> frame_record_worker_controller_;
		worker::ThreadWorkerController<worker::ChartRecordWorker> chart_record_worker_controller_;

		worker::ThreadWorkerController<worker::BatchGPIBWorker> batch_gpib_worker_controller_;

		worker::ThreadWorkerController<worker::InformationWorker> info_worker_controller_;

		worker::ThreadWorkerController<worker::ComputeWorker> compute_worker_controller_;
		std::atomic<std::shared_ptr<ICompute>> compute_pipe_{ nullptr };

		/*! \{ \name Frames queue (GPU) */
		std::atomic<std::shared_ptr<Queue>> gpu_input_queue_{ nullptr };
		std::atomic<std::shared_ptr<Queue>> gpu_output_queue_{ nullptr };
		/*! \} */

		/*! \brief Common compute descriptor shared between CLI/GUI and the
		 * Pipe. */
		ComputeDescriptor cd_;
	};
}

#include "holovibes.hxx"