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
#include "holo_file.hh"
#include "MainWindow.hh"
#include "cuda_memory.cuh"
#include "config.hh"

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
		FileType file_type,
		bool load_file_in_gpu,
		QProgressBar *reader_progress_bar,
		gui::MainWindow *main_window)
		: IThreadInput()
		, file_src_(file_src)
		, fd_(fd)
		, frame_size_(fd_.frame_size())
		, loop_(loop)
		, fps_(fps)
		, cur_frame_id_(first_frame_id - 1) // -1 because in ui frame start at 1
		, first_frame_id_(first_frame_id - 1)
		, last_frame_id_(last_frame_id - 1)
		, dst_queue_(input)
		, file_type_(file_type)
		, frame_annotation_size_(0)
		, load_file_in_gpu_(load_file_in_gpu)
		, progress_bar_refresh_interval_(fps / progress_bar_refresh_frequency_)
		, reader_progress_bar_(reader_progress_bar)
		, main_window_(main_window)
		, thread_(&ThreadReader::thread_proc, this)
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

		// Specific values for cine file
		if (file_type_ == FileType::CINE)
		{
			// Cine file format put an extra 8 bits header for every image
			frame_size_ += 8;
			frame_annotation_size_ = 8;
		}
	}

	FILE* ThreadReader::init_file(fpos_t* start_pos)
	{
		FILE* file = nullptr;

		fopen_s(&file, file_src_.c_str(), "rb");
		if (file == nullptr)
		{
			LOG_ERROR("[READER] unable to open file: " + file_src_);
			return nullptr;
		}

		// FIXME: Use the InputFile class
		unsigned int offset = 0;
		if (file_type_ == FileType::CINE)
		{
			// we look were the data is.
			offset = offset_cine_first_image(file);
		}
		else if (file_type_ == FileType::HOLO)
			offset = sizeof(HoloFile::Header);

		// Compute position of the first frame
		*start_pos = offset + frame_size_ * first_frame_id_;

		// failure on setting the position in the file
		if (std::fsetpos(file, start_pos) != 0)
		{
			LOG_ERROR("[READER] unable to read/open file: " + file_src_);
			return nullptr;
		}

		return file;
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
		if (load_file_in_gpu_) // TODO: When File class is available use the function
		 	buffer_nb_frames = last_frame_id_ + 1;
		else
		 	buffer_nb_frames = global::global_config.file_buffer_size;

		size_t buffer_size = frame_size_ * buffer_nb_frames;

		fpos_t start_position;
		FILE* file = init_file(&start_position);
		if (!file)
		{
			LOG_ERROR("[READER] unable to open file: " + file_src_);
			return;
		}

		// Init buffers
		char* cpu_buffer;
		cudaError_t error_code = cudaMallocHost(&cpu_buffer, sizeof(char) * buffer_size);
		if (error_code != cudaSuccess)
		{
			if (!load_file_in_gpu_)
				LOG_ERROR("[READER] Not enough CPU RAM to read file");
			else
				LOG_ERROR("[READER] Not enough CPU RAM to read file (consider disabling \"Load file in GPU\" option)");
			std::fclose(file);
			return;
		}

		char* gpu_buffer = nullptr;
		error_code = cudaMalloc((void**)&gpu_buffer, sizeof(char) * buffer_size);
		if (error_code != cudaSuccess)
		{
			if (!load_file_in_gpu_)
				LOG_ERROR("[READER] Not enough GPU DRAM to read file");
			else
				LOG_ERROR("[READER] Not enough GPU DRAM to read file (consider disabling \"Load file in GPU\" option)");
			cudaSafeCall(cudaFreeHost(cpu_buffer));
			std::fclose(file);
			return;
		}

		std::atomic<uint> nb_frames_one_second = 0;
		ThreadTimer thread_timer(nb_frames_one_second);

		FpsHandler fps_handler(fps_);
		fps_handler.begin();

		progress_bar_frame_counter_ = 0;

		if (load_file_in_gpu_)
			read_file_in_gpu(cpu_buffer, gpu_buffer, buffer_size, file, fps_handler, thread_timer, nb_frames_one_second);
		else
			read_file_batch(cpu_buffer, gpu_buffer, buffer_size, file, &start_position, fps_handler, thread_timer, nb_frames_one_second);

		std::fclose(file);
		// Free memory
		cudaSafeCall(cudaFreeHost(cpu_buffer));
		cudaXFree(gpu_buffer);
	}

	void ThreadReader::read_file_in_gpu(char* cpu_buffer,
									    char* gpu_buffer,
									    size_t buffer_size,
									    FILE* file,
									    FpsHandler& fps_handler,
									    ThreadTimer& thread_timer,
									    std::atomic<uint>& nb_frames_one_second)
	{
		// Read and copy the entire file
		size_t frames_read = read_copy_file(cpu_buffer, gpu_buffer, buffer_size, file);

		while (!stop_requested_)
		{
			enqueue_loop(gpu_buffer, frames_read, fps_handler, nb_frames_one_second);
			handle_last_frame();
		}
	}

	void ThreadReader::read_file_batch(char* cpu_buffer,
									   char* gpu_buffer,
									   size_t buffer_size,
									   FILE* file,
									   fpos_t* start_pos,
									   FpsHandler& fps_handler,
									   ThreadTimer& thread_timer,
									   std::atomic<uint>& nb_frames_one_second)
	{
		// Read the entire file by bactch
		while(!stop_requested_)
		{
			// Read batch in cpu and copy it to gpu
			size_t frames_read = read_copy_file(cpu_buffer, gpu_buffer, buffer_size, file);

			// Enqueue the batch frames one by one into the destination queue
			enqueue_loop(gpu_buffer, frames_read, fps_handler, nb_frames_one_second);

			// Reset to the first frame if needed
			handle_last_frame(file, start_pos);
		}
	}

	size_t ThreadReader::read_copy_file(char* cpu_buffer,
										char* gpu_buffer,
										size_t buffer_size,
										FILE* file)
	{
		// Read
		const size_t bytes_read = std::fread(cpu_buffer, sizeof(char), buffer_size, file);
		const size_t frames_read = bytes_read / frame_size_;

		// Memcopy in the gpu buffer
		cudaXMemcpy(gpu_buffer, cpu_buffer, bytes_read * sizeof(char), cudaMemcpyHostToDevice);

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

	void ThreadReader::handle_last_frame(FILE* file, fpos_t* start_pos)
	{
		// Reset to the first frame
		if (cur_frame_id_ > last_frame_id_)
		{
			if (loop_)
			{
				cur_frame_id_ = first_frame_id_;
				if (file && start_pos) // No start pos if loaded in gpu
					std::fsetpos(file, start_pos);
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

	long int ThreadReader::offset_cine_first_image(FILE *file)
	{
		char			buffer[44];
		unsigned int	offset_to_ptr = 0;
		fpos_t			off = 0;
		long int		offset_to_image = 0;

		/*Reading the whole cine file header*/
		if (std::fread(buffer, 1, 44, file) != 44)
			return 0;
		/*Reading OffImageOffsets for offset to POINTERS TO IMAGES*/
		std::memcpy(&offset_to_ptr, (buffer + 32), sizeof(int));
		/*Reading offset of the first image*/
		off = offset_to_ptr;
		std::fsetpos(file, &off);
		if (std::fread(&offset_to_image, 1, sizeof(long int), file) != sizeof(long int))
			return 0;
		return offset_to_image;
	}
}