# include "thread_reader.hh"
# include <fstream>
# include <Windows.h>

#include <chrono>

namespace holovibes
{
  ThreadReader::ThreadReader(std::string file_src
    , ThreadReader::FrameDescriptor frame_desc
    , bool loop
    , unsigned int fps
    , unsigned int spanStart
    , unsigned int spanEnd
    , Queue& input)
    : IThreadInput()
    , file_src_(file_src)
    , frame_desc_(frame_desc.desc)
    , desc_(frame_desc)
    , loop_(loop)
    , fps_(fps)
    , frameId_(0)
    , spanStart_(spanStart)
    , spanEnd_(spanEnd)
    , queue_(input)
    , thread_(&ThreadReader::thread_proc, this)
  {
  }

  void	ThreadReader::thread_proc()
  {
    std::ifstream	ifs(file_src_, std::istream::in | std::ifstream::binary);
    std::streampos pos;
    unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
    char* buffer = new char[frame_size * NBR];
    unsigned int nbr_stored = 0;
    unsigned int act_frame = 0;

    try
    {
      if (!ifs.is_open())
        throw std::runtime_error("[READER] unable to read/open file: " + file_src_);
      while (++frameId_ < spanStart_)
        ifs.read(buffer, frame_size);
      pos = ifs.tellg();

      while (!stop_requested_)
      {
        if (ifs.good() && frameId_ <= spanEnd_)
        {
          if (act_frame >= nbr_stored)
          {
            ifs.read(buffer, frame_size * NBR);
            nbr_stored = ifs.gcount() / frame_size;
            act_frame = 0;
          }
          queue_.enqueue(buffer + act_frame * frame_size, cudaMemcpyHostToDevice);
          ++frameId_;
          ++act_frame;
          Sleep(1000 / fps_);
        }
        else if (loop_)
        {
          ifs.clear();
          ifs.seekg(pos);
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
    if (ifs.is_open())
      ifs.close();
    stop_requested_ = true;
    delete[] buffer;
  }

  ThreadReader::~ThreadReader()
  {
    stop_requested_ = true;
    if (thread_.joinable())
      thread_.join();
  }
}