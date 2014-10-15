#include "stdafx.h"
#include "recorder.hh"

#include <exception>

namespace holovibes
{
  Recorder::Recorder(Queue *queue, std::string path, unsigned set_size)
    :path_(path)
  {
#ifndef _DEBUG
    if (check_overwrite())
      throw std::exception("overwriting an existing file");
#endif /* Overwrite is useful while debugging. */
    if (queue->get_max_elts() <= set_size)
      set_size_ = 1;
    else
      set_size_ = set_size;
    buffer_ = queue;
    if (fopen_s(&fd_, path.c_str(), "w+b") != 0)
      throw std::exception("can not open the specified file for writing data");
  }

  void Recorder::record()
  {
    if (buffer_->get_current_elts() >= set_size_) // we record them to the drive
    {
      size_t written = contigous_image();
      size_t elt_written = fwrite(buffer_->dequeue(written), buffer_->get_size(), written, fd_);
      if (elt_written != written)
        throw std::exception("one or more images were not correctly saved, the save file is corrupted");
    }
  }
  size_t Recorder::contigous_image()
  {
    if (buffer_->get_max_elts() - buffer_->get_start_index() >= buffer_->get_current_elts())
      return buffer_->get_current_elts();
    else
    {
      return buffer_->get_max_elts() - buffer_->get_start_index();
    }
  }
  bool Recorder::check_overwrite()
  {
    FILE *fdtmp;
    if (fopen_s(&fdtmp, path_.c_str(), "r") == 0)
      return true;
    return false;
  }
  Recorder::~Recorder()
  {
    if (fd_)
      fclose(fd_);
  }
}