#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "stdafx.h"
#include "recorder.hh"


namespace holovibes
{
  Recorder::Recorder(queue::Queue *queue, std::string path, int set_size)
  {
    if (queue->get_max_elts() <= set_size)
      set_size_ = 1;
    else
      set_size_ = set_size;
    buffer_ = queue;
    //fopen_s(&fd_,path.c_str(), "w+");
    fd_ = fopen(path.c_str(), "w+");
  }

  void Recorder::record()
  {
    if (buffer_->get_current_elts() >= set_size_) // we record them to the drive
    {
      std::cout << "record" << std::endl;
      size_t written = contigous_image();
      std::cout << "written" << written << std::endl;
      fwrite(buffer_->dequeue(written), buffer_->get_size(), written, fd_);
    }
  }
  unsigned int Recorder::contigous_image()
  {
    if (buffer_->get_max_elts() - buffer_->get_start_index() >= buffer_->get_current_elts())
      return buffer_->get_current_elts();
    else
    {
      return buffer_->get_max_elts() - buffer_->get_start_index();
    }
      
  }
  Recorder::~Recorder()
  {
    if (fd_)
    fclose(fd_);
  }
}