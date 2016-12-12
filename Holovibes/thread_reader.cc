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

namespace holovibes
{
<<<<<<< 0da8e21add50d421ac96c02bd36a59a8e48dfac3
  ThreadReader::ThreadReader(std::string file_src,
    camera::FrameDescriptor& new_frame_desc,
	camera::FrameDescriptor& frame_desc,
    bool loop,
    unsigned int fps,
    unsigned int spanStart,
    unsigned int spanEnd,
    Queue& input,
	bool is_cine_file,
	holovibes::Holovibes& holovibes)
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
    , thread_(&ThreadReader::thread_proc, this)
  {
    gui::InfoManager::get_manager()->update_info("ImgSource", "File");
  }

  void ThreadReader::thread_proc()
  {
	  if (is_cine_file_ == true)
		  proc_cine_file();
	  else
		  proc_default();
  }

  void	ThreadReader::proc_default()
  {
	  unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
	  unsigned int real_frame_size = real_frame_desc_.width * real_frame_desc_.height * real_frame_desc_.depth;
	  unsigned int elts_max_nbr = global::global_config.reader_buf_max_size;
	  char*        buffer = NULL;
	  char*		   real_buffer = NULL;
	  unsigned int nbr_stored = 0;
	  unsigned int act_frame = 0;
	  FILE*   file = nullptr;
	  fpos_t  pos = 0;
	  size_t  length = 0;

	  cudaMallocHost(&buffer, frame_size * elts_max_nbr);
	  cudaMallocHost(&real_buffer, real_frame_size);
	  try
	  {
		  fopen_s(&file, file_src_.c_str(), "rb");
		  if (!file)
			  throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
		  pos = frame_size * (spanStart_ - 1);
		  std::fsetpos(file, &pos);
		  cudaMemset(real_buffer, 0, real_frame_desc_.frame_size());
		  while (!stop_requested_)
		  {
			  if (!std::feof(file) && frameId_ <= spanEnd_)
			  {
				  if (act_frame >= nbr_stored)
				  {			  
					  length = std::fread(buffer, 1, frame_size * elts_max_nbr, file);
					  nbr_stored = length / frame_size;
					  act_frame = 0;
				  }
				  if (real_frame_desc_.width == frame_desc_.width && real_frame_desc_.height == frame_desc_.height)
					  queue_.enqueue(buffer + act_frame * frame_size, cudaMemcpyHostToDevice);
				  else
				  {
					  buffer_size_conversion(real_buffer
						  , buffer + act_frame * frame_size
						  , real_frame_desc_
						  , frame_desc_);
					  queue_.enqueue(real_buffer, cudaMemcpyHostToDevice);
				  }
				  ++frameId_;
				  ++act_frame;
				  Sleep(1000 / fps_);
			  }
			  else if (loop_)
			  {
				  std::clearerr(file);
				  std::fsetpos(file, &pos);
				  frameId_ = spanStart_;
				  int offset = elts_max_nbr - length;
			  }
			  else
				  stop_requested_ = true;
		  }
	  }
	  catch (std::runtime_error& e)
	  {
		  std::cout << e.what() << std::endl;
	  }
	  if (file)
	  {
		  std::fclose(file);
		  file = nullptr;
	  }
	  stop_requested_ = true;
	  cudaFreeHost(buffer);
	  cudaFreeHost(real_buffer);
  }

  void	ThreadReader::proc_cine_file()
  {
	  unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
	  unsigned int real_frame_size = real_frame_desc_.width * real_frame_desc_.height * real_frame_desc_.depth;
	  unsigned int elts_max_nbr = global::global_config.reader_buf_max_size;
	  char*        buffer = NULL;
	  char*        real_buffer = NULL;
	  unsigned int nbr_stored = 0;
	  unsigned int act_frame = 0;
	  FILE*   file = nullptr;
	  fpos_t  pos = 0;

	  cudaMallocHost(&buffer, (frame_size + 8) * elts_max_nbr);
	  cudaMallocHost(&real_buffer, real_frame_size);
	  try
	  {
		  fopen_s(&file, file_src_.c_str(), "rb");
		  if (!file)
			  throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
=======
	ThreadReader::ThreadReader(std::string file_src,
		camera::FrameDescriptor& new_frame_desc,
		camera::FrameDescriptor& frame_desc,
		bool loop,
		unsigned int fps,
		unsigned int spanStart,
		unsigned int spanEnd,
		Queue& input,
		bool is_cine_file,
		holovibes::Holovibes& holovibes)
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
		gui::InfoManager::get_manager()->update_info("ImgSource", "File");
	}

	void ThreadReader::thread_proc()
	{
		clock_t deltaTime = 0;
		unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
		unsigned int resize_frame_size = real_frame_desc_.width * real_frame_desc_.height * real_frame_desc_.depth;
		unsigned int elts_max_nbr = global::global_config.reader_buf_max_size;
		char* buffer = nullptr;
		char* resize_buffer = nullptr;
		if (real_frame_desc_.width != frame_desc_.width || real_frame_desc_.height != frame_desc_.height)
			cudaMallocHost(&resize_buffer, resize_frame_size);
		FILE*   file = nullptr;
		unsigned int offset = 0;
		try
		{
			fopen_s(&file, file_src_.c_str(), "rb");
			if (is_cine_file_)
			{
				offset = offset_cine_first_image(file);
				frame_size += 8;
			}
			cudaMallocHost(&buffer, frame_size * elts_max_nbr);
			fpos_t pos = offset + frame_size * (spanStart_ - 1);
			std::fsetpos(file, &pos);
			if (!file)
				throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
			clock_t deltaTime = 0;
			while (!stop_requested_)
			{
				clock_t beginFrame = clock();
				reader_loop(file, buffer, resize_buffer, frame_size, elts_max_nbr, pos);
				clock_t endFrame = clock();
				deltaTime = endFrame - beginFrame;
				double timelaps = clockToMilliseconds(endFrame - beginFrame);
				if (timelaps < 1000)
					Sleep((1000 - timelaps) / fps_);
			}
		}
		catch (std::runtime_error& e)
		{
			std::cout << e.what() << std::endl;
		}
		if (file)
			std::fclose(file);
		stop_requested_ = true;
		cudaFreeHost(buffer);
		cudaFreeHost(resize_buffer);
	}
>>>>>>> Update : thread_reader is now implemented in a cleaner way

	void ThreadReader::reader_loop(
		FILE* file,
		char* buffer,
		char* resize_buffer,
		const unsigned int& frame_size,
		const unsigned int& elts_max_nbr,
		fpos_t pos)
	{
		unsigned int offset = 0;
		if (is_cine_file_)
			offset = 8;

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
				return;
			}
		}
		if (act_frame_ >= nbr_stored_)
		{
			size_t length = std::fread(buffer, 1, frame_size * elts_max_nbr, file);
			nbr_stored_ = length / frame_size;
			act_frame_ = 0;
		}
		if (real_frame_desc_.width == frame_desc_.width && real_frame_desc_.height == frame_desc_.height)
			queue_.enqueue(buffer + offset + act_frame_ * frame_size, cudaMemcpyHostToDevice);
		else
		{
			buffer_size_conversion(resize_buffer
				, buffer + offset + act_frame_ * frame_size
				, real_frame_desc_
				, frame_desc_);
			queue_.enqueue(resize_buffer, cudaMemcpyHostToDevice);
		}
		++frameId_;
		++act_frame_;
	}

  ThreadReader::~ThreadReader()
  {
    stop_requested_ = true;

    if (thread_.joinable())
      thread_.join();
    gui::InfoManager::get_manager()->update_info("ImgSource", "none");
  }

  const camera::FrameDescriptor& ThreadReader::get_frame_descriptor() const
  {
    return frame_desc_;
  }

  long int ThreadReader::offset_cine_first_image(FILE *file)
  {
	  long int		length = 0;
	  char			buffer[44];
	  unsigned int	offset_to_ptr = 0;
	  fpos_t		off = 0;
	  long int		offset_to_image = 0;

	  if ((length = std::fread(buffer, 1, 44, file)) =! 44)
		  return (0);
	  std::memcpy(&offset_to_ptr, (buffer + 32), sizeof(int));
	  off = offset_to_ptr;
	  std::fsetpos(file, &off);
	  if ((length = std::fread(&offset_to_image, 1, sizeof(long int), file)) =! sizeof(long int))
		  return (0);
	  return (offset_to_image);
  }
}