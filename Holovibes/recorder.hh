#ifndef RECORDER_HH
# define RECORDER_HH

#include <stdio.h>
#include "queue.hh"
namespace holovibes
{
  class Recorder
  {
  public:
    Recorder(queue::Queue *queue, std::string path, unsigned int set_size);
    void record();
    unsigned int contigous_image();
    ~Recorder();

  private:
    queue::Queue *buffer_;
    FILE *fd_;
    unsigned int set_size_;
  };
}

#endif