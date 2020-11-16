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

#include <cuda.h>
#include <algorithm>
#include <atomic>

#include "thread_reader.hh"
#include "tools_conversion.cuh"
#include "info_manager.hh"
#include "config.hh"
#include "queue.hh"
#include "holovibes.hh"
#include "tools.hh"
#include "MainWindow.hh"
#include "cuda_memory.cuh"
#include "config.hh"
#include "input_file_handler.hh"

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace holovibes
{
	ThreadReader::ThreadReader(std::string file_src,
		camera::FrameDescriptor& fd,
		SquareInputMode mode,
		bool loop,
		unsigned int fps,
		size_t first_frame_id,
		size_t last_frame_id,
		Queue& input,
		bool load_file_in_gpu,
		QProgressBar *reader_progress_bar,
		gui::MainWindow *main_window)
		: IThreadInput()
		, file_src_(file_src)
		, fd_(fd)
		, frame_size_(io_files::InputFileHandler::get_frame_descriptor().frame_size())
		, loop_(loop)
		, fps_(fps)
		, cur_frame_id_(first_frame_id - 1) // -1 because in ui frame start at 1
		, first_frame_id_(first_frame_id - 1)
		, last_frame_id_(last_frame_id - 1)
		, dst_queue_(input)
		, frame_annotation_size_(io_files::InputFileHandler::get_frame_annotation_size())
		, load_file_in_gpu_(load_file_in_gpu)
		, progress_bar_refresh_interval_(std::max(1., fps / progress_bar_refresh_frequency_))
		, reader_progress_bar_(reader_progress_bar)
		, main_window_(main_window)
	{
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "File");
		dst_queue_.set_square_input_mode(mode);
		std::string input_descriptor_info = std::to_string(fd_.width)
			+ std::string("x")
			+ std::to_string(fd_.height)
			+ std::string(" - ")
			+ std::to_string(static_cast<int>(fd_.depth * 8))
			+ std::string("bit");
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::INPUT_SOURCE, "InputFormat", input_descriptor_info);
		reader_progress_bar->setMaximum(last_frame_id);
		reader_progress_bar->show();

		frame_size_ += frame_annotation_size_;

		thread_ = std::thread(&ThreadReader::thread_proc, this);
	}

	/*! \class FpsHandler
    * \brief Private class used to handle input fps
	*/
	class FpsHandler
	{
		public:
			FpsHandler(uint fps)
			: enqueue_interval_((1 / static_cast<double>(fps)))
			{}

			/*! \brief Begin the process of fps handling. */
			void begin()
			{
				begin_time_ = std::chrono::high_resolution_clock::now();
			}

			/*! \brief Wait the correct time to simulate fps.
			**
			** Between each frame enqueue, the waiting duration should be enqueue_interval_
			** However the real waiting duration might be longer than the theoretical one (due to descheduling)
			** To cope with this issue, we compute the wasted time in order to take it into account for the next enqueue
			** By doing so, the correct enqueuing time is computed, not doing so would create a lag
			**/
			void wait()
			{
				/* end_time should only be being_time + enqueue_interval_ aka the time point for the next enqueue
				** However the wasted_time is substracted to get the correct next enqueue time point
				*/
				auto end_time = (begin_time_ + enqueue_interval_) - wasted_time_;

				// Wait until the next enqueue time point is reached
				while (std::chrono::high_resolution_clock::now() < end_time)
					continue;

				/* Wait is done, it might have been too long (descheduling...)
				**
				** Set the begin_time (now) for the next enqueue
				** And compute the wasted time (real time point - theoretical time point)
				*/
				auto now =  std::chrono::high_resolution_clock::now();
				wasted_time_ = now - end_time;
				begin_time_ = now;
			}

		private:
			/*! \brief Theoretical time between 2 enqueues/waits */
			std::chrono::duration<double> enqueue_interval_;

			/*! \brief Begin time point of the wait */
			std::chrono::steady_clock::time_point begin_time_;

			/*! \brief Time wasted in last wait (if waiting was too long) */
			std::chrono::duration<double> wasted_time_{0};
	};

	void ThreadReader::thread_proc()
	{
		SetThreadPriority(thread_.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);

		size_t buffer_nb_frames;

		if (load_file_in_gpu_)
		 	buffer_nb_frames = io_files::InputFileHandler::get_total_nb_frames();
		else
		 	buffer_nb_frames = global::global_config.file_buffer_size;

		size_t buffer_size = frame_size_ * buffer_nb_frames * sizeof(char);

		// Init buffers
		char* cpu_buffer;
		cudaError_t error_code = cudaMallocHost(&cpu_buffer, buffer_size);

		if (error_code != cudaSuccess)
		{
			if (!load_file_in_gpu_)
				LOG_ERROR("[READER] Not enough CPU RAM to read file");
			else
				LOG_ERROR("[READER] Not enough GPU DRAM to read file (consider disabling \"Load file in GPU\" option)");

			return;
		}

		char* gpu_buffer = nullptr;
		error_code = cudaMalloc((void**)&gpu_buffer, buffer_size);

		if (error_code != cudaSuccess)
		{
			if (!load_file_in_gpu_)
				LOG_ERROR("[READER] Not enough GPU DRAM to read file");
			else
				LOG_ERROR("[READER] Not enough GPU DRAM to read file (consider disabling \"Load file in GPU\" option)");

			cudaSafeCall(cudaFreeHost(cpu_buffer));
			return;
		}

		std::atomic<uint> nb_frames_one_second = 0;
		ThreadTimer thread_timer(nb_frames_one_second);

		FpsHandler fps_handler(fps_);
		fps_handler.begin();

		progress_bar_frame_counter_ = 0;

		try
		{
			io_files::InputFileHandler::set_pos_to_first_frame();

			if (load_file_in_gpu_)
				read_file_in_gpu(cpu_buffer, gpu_buffer, fps_handler, nb_frames_one_second);
			else
				read_file_batch(cpu_buffer, gpu_buffer, buffer_nb_frames, fps_handler, nb_frames_one_second);
		}
		catch (const io_files::FileException& e)
		{
			LOG_ERROR("[READER] " + std::string(e.what()));
		}

		// Free memory
		cudaSafeCall(cudaFreeHost(cpu_buffer));
		cudaXFree(gpu_buffer);
	}

	void ThreadReader::read_file_in_gpu(char* cpu_buffer,
									    char* gpu_buffer,
									    FpsHandler& fps_handler,
									    std::atomic<uint>& nb_frames_one_second)
	{
		// Read and copy the entire file
		size_t frames_to_read = io_files::InputFileHandler::get_total_nb_frames();
		size_t frames_read = read_copy_file(cpu_buffer, gpu_buffer, frames_to_read);

		while (!stop_requested_)
		{
			enqueue_loop(gpu_buffer, frames_read, fps_handler, nb_frames_one_second);
			handle_last_frame();
		}
	}

	void ThreadReader::read_file_batch(char* cpu_buffer,
									   char* gpu_buffer,
									   size_t frames_to_read,
									   FpsHandler& fps_handler,
									   std::atomic<uint>& nb_frames_one_second)
	{
		// Read the entire file by bactch
		while(!stop_requested_)
		{
			// Read batch in cpu and copy it to gpu
			size_t frames_read = read_copy_file(cpu_buffer, gpu_buffer, frames_to_read);

			// Enqueue the batch frames one by one into the destination queue
			enqueue_loop(gpu_buffer, frames_read, fps_handler, nb_frames_one_second);

			// Reset to the first frame if needed
			handle_last_frame();
		}
	}

	size_t ThreadReader::read_copy_file(char* cpu_buffer,
										char* gpu_buffer,
									    size_t frames_to_read)
	{
		// Read
		size_t frames_read = 0;

		try
		{
			frames_read = io_files::InputFileHandler::read_frames(cpu_buffer, frames_to_read);
		}
		catch (const io_files::FileException& e)
		{
			LOG_WARN(e.what());
		}

		size_t frames_total_size = frames_read * frame_size_;

		// Memcopy in the gpu buffer
		cudaXMemcpy(gpu_buffer, cpu_buffer, frames_total_size, cudaMemcpyHostToDevice);

		return frames_read;
	}

	void ThreadReader::enqueue_loop(char* gpu_buffer,
									size_t nb_frames_to_enqueue,
									FpsHandler& fps_handler,
									std::atomic<uint>& nb_frames_one_second)
	{
		size_t frames_enqueued = 0;
		while (frames_enqueued < nb_frames_to_enqueue && cur_frame_id_ <= last_frame_id_ && !stop_requested_)
		{
			fps_handler.wait();
			if (!dst_queue_.enqueue(gpu_buffer + frames_enqueued * frame_size_ + frame_annotation_size_, cudaMemcpyDeviceToDevice))
			{
				LOG_ERROR("[READER] Cannot enqueue a read frame");
				return;
			}

			++cur_frame_id_;
			++frames_enqueued;
			++nb_frames_one_second;
			++progress_bar_frame_counter_;

			// Update GUI
			// Updates aren't always done because it would slow down the program
			if (progress_bar_frame_counter_ == progress_bar_refresh_interval_ && main_window_)
			{
				main_window_->update_file_reader_index(cur_frame_id_ - first_frame_id_);
				progress_bar_frame_counter_ = 0;
			}
		}
	}

	void ThreadReader::handle_last_frame()
	{
		// Reset to the first frame
		if (cur_frame_id_ > last_frame_id_)
		{
			if (loop_)
			{
				cur_frame_id_ = first_frame_id_;
				io_files::InputFileHandler::set_pos_to_first_frame();
				emit at_begin();
				// continue
			}
			else
				stop_requested_ = true; // break
		}
	}

	ThreadReader::~ThreadReader()
	{
		stop_requested_ = true;
		if (reader_progress_bar_)
			reader_progress_bar_->hide();

		while (!thread_.joinable())
			continue;
		thread_.join();
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "none");
	}

	const camera::FrameDescriptor& ThreadReader::get_input_fd() const
	{
		return fd_;
	}

	const camera::FrameDescriptor& ThreadReader::get_queue_fd() const
	{
		return dst_queue_.get_fd();
	}
}
