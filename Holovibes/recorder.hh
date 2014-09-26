#ifndef RECORDER_HH
# define RECORDER_HH

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include "queue.hh"
namespace recorder
{
  class Recorder
  {
  public:
    Recorder(queue::Queue *queue, std::string path, int set_size);
    void record();
    unsigned int contigous_image();
    ~Recorder();

  private:
    queue::Queue *buffer_;
    FILE *fd_;
    int set_size_;

  };
}

#endif