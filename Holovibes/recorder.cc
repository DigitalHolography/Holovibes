#include "recorder.hh"

namespace recorder
{
  Recorder::Recorder(queue::Queue *queue, std::string path, int set_size)
  {
    //if (buffer_ && buffer_->get_max_elts() > 
    set_size_ = set_size;
    buffer_ = queue;
    fd_ = fopen(path.c_str(), "w+");
  }

  void Recorder::record()
  {
    int written = 0;
    if (buffer_->get_current_elts() >= set_size_) // we record them to the drive
    {
      written = contigous_image();
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
      std::cout << "BOOOM" << std::endl;
      return buffer_->get_max_elts() - buffer_->get_start_index();
    }
      
  }
  Recorder::~Recorder()
  {
  }
}