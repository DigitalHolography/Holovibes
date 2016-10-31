# include <fstream>
# include <Windows.h>
# include <chrono>

# include "tools_conversion.cuh"
# include "info_manager.hh"
# include "config.hh"
# include "thread_reader.hh"
# include "queue.hh"
# include "holovibes.hh"

namespace holovibes
{
  ThreadReader::ThreadReader(std::string file_src,
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
		  proc_8_16_32();
  }

  void	ThreadReader::proc_8_16_32()
  {
	  unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
	  unsigned int elts_max_nbr = global::global_config.input_queue_max_size;
	  char*        buffer;
	  unsigned int nbr_stored = 0;
	  unsigned int act_frame = 0;
	  FILE*   file = nullptr;
	  fpos_t  pos = 0;
	  size_t  length = 0;

	  cudaMallocHost(&buffer, frame_size * elts_max_nbr);
	  try
	  {
		  fopen_s(&file, file_src_.c_str(), "rb");
		  if (!file)
			  throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
		  pos = frame_size * (spanStart_ - 1);
		  std::fsetpos(file, &pos);
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
				  queue_.enqueue(buffer + act_frame * frame_size, cudaMemcpyHostToDevice);
				  ++frameId_;
				  ++act_frame;
				  Sleep(1000 / fps_);
			  }
			  else if (loop_)
			  {
				  std::clearerr(file);
				  std::fsetpos(file, &pos);
				  frameId_ = spanStart_;
				  act_frame = 0;
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
  }

  void	ThreadReader::proc_cine_file()
  {
	  unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
	  unsigned int elts_max_nbr = global::global_config.input_queue_max_size;
	  char*        buffer;
	  unsigned int nbr_stored = 0;
	  unsigned int act_frame = 0;
	  FILE*   file = nullptr;
	  fpos_t  pos = 0;
	  size_t  length = 0;
	  size_t  offset = 0;

	  cudaMallocHost(&buffer, (frame_size + 8) * elts_max_nbr);
	  try
	  {
		  fopen_s(&file, file_src_.c_str(), "rb");
		  if (!file)
			  throw std::runtime_error("[READER] unable to read/open file: " + file_src_);

		  offset = offset_cine_first_image(file);
		  pos = offset + (frame_size + 8) * (spanStart_ - 1);
		  std::fsetpos(file, &pos);
		  while (!stop_requested_)
		  {
			  if (!std::feof(file) && frameId_ <= spanEnd_)
			  {
				  if (act_frame >= nbr_stored)
				  {
					  length = std::fread(buffer, 1, (frame_size + 8) * elts_max_nbr, file);
					  nbr_stored = length / (frame_size + 8);
					  act_frame = 0;
				  }
				  queue_.enqueue(buffer + 8 * (act_frame + 1) + act_frame * frame_size, cudaMemcpyHostToDevice);
				  ++frameId_;
				  ++act_frame;
				  Sleep(1000 / fps_);
			  }
			  else if (loop_)
			  {
				  std::clearerr(file);
				  std::fsetpos(file, &pos);
				  frameId_ = spanStart_;
				  act_frame = 0;
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