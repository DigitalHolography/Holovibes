# include "thread_reader.hh"
# include <fstream>
# include <Windows.h>
# include <chrono>

#include "info_manager.hh"
#include "config.hh"

namespace holovibes
{
  ThreadReader::ThreadReader(std::string file_src,
    ThreadReader::FrameDescriptor frame_desc,
    bool loop,
    unsigned int fps,
    unsigned int spanStart,
    unsigned int spanEnd,
    Queue& input)
    : IThreadInput()
    , file_src_(file_src)
    , frame_desc_(frame_desc.desc)
    , desc_(frame_desc)
    , loop_(loop)
    , fps_(fps)
    , frameId_(spanStart)
    , spanStart_(spanStart)
    , spanEnd_(spanEnd)
    , queue_(input)
    , thread_(&ThreadReader::thread_proc, this)
  {
    gui::InfoManager::get_manager()->update_info("ImgSource", "File");
  }

  void ThreadReader::thread_proc()
  {
    unsigned int frame_size = frame_desc_.width * frame_desc_.height * frame_desc_.depth;
    unsigned int elts_max_nbr = global::global_config.input_queue_max_size;
    char*        buffer;
    cudaMallocHost(&buffer, frame_size * elts_max_nbr);
    unsigned int nbr_stored = 0;
    unsigned int act_frame = 0;
    FILE*   file = nullptr;
    fpos_t  pos = frame_size * (spanStart_ - 1);

    try
    {
      fopen_s(&file, file_src_.c_str(), "rb");
      if (!file)
        throw std::runtime_error("[READER] unable to read/open file: " + file_src_);

      std::fsetpos(file, &pos);

      while (!stop_requested_)
      {
        if (!std::feof(file) && frameId_ <= spanEnd_)
        {
          if (act_frame >= nbr_stored)
          {
            size_t length = std::fread(buffer, 1, frame_size * elts_max_nbr, file);
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

  ThreadReader::~ThreadReader()
  {
    stop_requested_ = true;

    if (thread_.joinable())
      thread_.join();
    gui::InfoManager::get_manager()->update_info("ImgSource", "none");
  }
}