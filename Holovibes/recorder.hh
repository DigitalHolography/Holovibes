#ifndef RECORDER_HH
# define RECORDER_HH

#include <stdio.h>

#include "queue.hh"
#include "error_handler.hh"

namespace holovibes
{
  class Recorder
  {
  public:
    Recorder(Queue *queue, std::string path, unsigned int set_size);
    void record();
    bool check_overwrite();
    size_t contigous_image();
    ~Recorder();

  private:
    std::string path_;
    Queue *buffer_;
    FILE *fd_;
    unsigned int set_size_;
  };
}

#endif