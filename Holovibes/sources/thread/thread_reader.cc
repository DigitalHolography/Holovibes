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

# include <fstream>
# include <Windows.h>
# include <chrono>

# include "tools_conversion.cuh"
# include "info_manager.hh"
# include "config.hh"
# include "thread_reader.hh"
# include "queue.hh"
# include "holovibes.hh"
# include "tools.hh"

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

namespace holovibes
{
	ThreadReader::ThreadReader(std::string file_src,
		camera::FrameDescriptor& new_frame_desc,
		camera::FrameDescriptor& frame_desc,
		bool loop,
		unsigned int fps,
		unsigned int spanStart,
		unsigned int spanEnd,
		Queue& input,
		bool is_cine_file,
		Holovibes& holovibes)
		: IThreadInput()
		, file_src_(file_src)
		, frame_desc_(frame_desc)
		, real_frame_desc_(new_frame_desc)
		, loop_(loop)
		, fps_(fps)
		, frameId_(spanStart)
		, spanStart_(spanStart)
		, spanEnd_(spanEnd)
		, queue_(input)
		, is_cine_file_(is_cine_file)
		, holovibes_(holovibes)
		, act_frame_(0)
		, nbr_stored_(0)
		, thread_(&ThreadReader::thread_proc, this)
	{
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "File");
		auto fd = get_frame_descriptor();
		std::string input_descriptor_info = std::to_string(fd.width)
			+ std::string("x")
			+ std::to_string(fd.height)
			+ std::string(" - ")
			+ std::to_string(static_cast<int>(fd.depth * 8))
			+ std::string("bit");
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::INPUT_SOURCE, "InputFormat", input_descriptor_info);
	}

	void ThreadReader::clear_memory(char **buffer, char **resize_buffer)
	{
		cudaFreeHost(*buffer);
		cudaFree(*resize_buffer);
		*buffer = nullptr;
		*resize_buffer = nullptr;
	}

	void ThreadReader::thread_proc()
	{
		unsigned int refresh_fps = fps_;
		unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
		unsigned int resize_frame_size = real_frame_desc_.width * real_frame_desc_.height * real_frame_desc_.depth;
		unsigned int elts_max_nbr = global::global_config.reader_buf_max_size;
		char* buffer = nullptr;
		char* resize_buffer = nullptr;
		if (real_frame_desc_.width != frame_desc_.width || real_frame_desc_.height != frame_desc_.height)
		{
			if (cudaMalloc(&resize_buffer, resize_frame_size) != CUDA_SUCCESS)
				throw std::runtime_error("[CUDA] : Memory allocation failed");
		}
		FILE*   file = nullptr;
		unsigned int offset = 0;
		const Clock::duration frame_frequency = std::chrono::microseconds(1000000 / fps_);
		try
		{
			fopen_s(&file, file_src_.c_str(), "rb");
			if (file == nullptr)
				throw std::runtime_error("[READER] unable to open file: " + file_src_);
			if (is_cine_file_)
			{
				// we look were the data is.
				offset = offset_cine_first_image(file);
				// Cine file format put an extra 8 bits header for every image
				frame_size += 8;
			}
			if (cudaMallocHost(&buffer, frame_size * elts_max_nbr) != CUDA_SUCCESS)
			{
				clear_memory(&buffer, &resize_buffer);
				throw std::runtime_error("[CUDA] : Memory allocation failed");
			}
			fpos_t pos = offset + frame_size * (spanStart_ - 1);
			std::fsetpos(file, &pos);
			if (!file)
			{
				clear_memory(&buffer, &resize_buffer);
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			}
			auto beginFrames = std::chrono::high_resolution_clock::now();
			auto next_game_tick = std::chrono::high_resolution_clock::now();
			while (!stop_requested_)
			{
				while (std::chrono::high_resolution_clock::now() > next_game_tick && !stop_requested_)
				{
					if (!reader_loop(file, buffer, resize_buffer, frame_size, elts_max_nbr, pos))
						stop_requested_ = true;
					next_game_tick += frame_frequency;
					if (--refresh_fps == 0)
					{
						auto endframes = std::chrono::high_resolution_clock::now();
						std::chrono::duration<float, std::milli> timelaps = endframes - beginFrames;
						auto manager = gui::InfoManager::get_manager();
						int fps = (fps_ / (timelaps.count() / 1000.0f));
						manager->insert_info(gui::InfoManager::InfoType::INPUT_FPS, "InputFps", std::to_string(fps) + std::string(" fps"));
						refresh_fps = fps_;
						beginFrames = std::chrono::high_resolution_clock::now();
					}
				}
			}
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
		if (file)
			std::fclose(file);
		stop_requested_ = true;
		clear_memory(&buffer, &resize_buffer);
	}

	bool ThreadReader::reader_loop(
		FILE* file,
		char* buffer,
		char* resize_buffer,
		const unsigned int& frame_size,
		const unsigned int& elts_max_nbr,
		fpos_t pos)
	{
		unsigned int cine_offset = 0;
		if (is_cine_file_)
			cine_offset = 8;

		if (std::feof(file) || frameId_ > spanEnd_)
		{
			if (loop_)
			{
				std::clearerr(file);
				std::fsetpos(file, &pos);
				frameId_ = spanStart_;
			}
			else
			{
				stop_requested_ = true;
				return true;
			}
		}
		if (act_frame_ >= nbr_stored_)
		{
			size_t length = std::fread(buffer, 1, frame_size * elts_max_nbr, file);
			nbr_stored_ = static_cast<unsigned int>(length) / frame_size;
			act_frame_ = 0;
		}
		if (real_frame_desc_.width == frame_desc_.width && real_frame_desc_.height == frame_desc_.height)
		{
			if (!queue_.enqueue(buffer + cine_offset + act_frame_ * frame_size, cudaMemcpyHostToDevice))
				return false;
		}
		else
		{
			buffer_size_conversion(resize_buffer
				, buffer + cine_offset + act_frame_ * frame_size
				, real_frame_desc_
				, frame_desc_);
			if (!queue_.enqueue(resize_buffer, cudaMemcpyDeviceToDevice))
				return false;
		}
		++frameId_;
		++act_frame_;
		return true;
	}

	ThreadReader::~ThreadReader()
	{
		stop_requested_ = true;

		while (!thread_.joinable())
			continue;
		thread_.join();
		gui::InfoManager::get_manager()->insert_info(gui::InfoManager::InfoType::IMG_SOURCE, "ImgSource", "none");
	}

	const camera::FrameDescriptor& ThreadReader::get_frame_descriptor() const
	{
		return frame_desc_;
	}

	long int ThreadReader::offset_cine_first_image(FILE *file)
	{
		size_t		length = 0;
		char			buffer[44];
		unsigned int	offset_to_ptr = 0;
		fpos_t		off = 0;
		long int		offset_to_image = 0;

		/*Reading the whole cine file header*/
		if ((length = std::fread(buffer, 1, 44, file)) != 44)
			return 0;
		/*Reading OffImageOffsets for offset to POINTERS TO IMAGES*/
		std::memcpy(&offset_to_ptr, (buffer + 32), sizeof(int));
		/*Reading offset of the first image*/
		off = offset_to_ptr;
		std::fsetpos(file, &off);
		if ((length = std::fread(&offset_to_image, 1, sizeof(long int), file)) != sizeof(long int))
			return 0;
		return offset_to_image;
	}
}