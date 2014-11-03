#include "stdafx.h"
#include "recorder.hh"

#include <exception>
#include <cassert>
#include <thread>
#include <iostream>

namespace holovibes
{
  Recorder::Recorder(
    Queue& queue,
    const std::string& filepath)
    : queue_(queue)
    , file_()
  {
#ifndef _DEBUG
    if (is_file_exist(filepath))
      throw std::exception("[RECORDER] error overwriting an existing file");
#endif /* Overwrite is useful while debugging. */

    file_.open(filepath, std::ios::binary | std::ios::trunc);
  }

  Recorder::~Recorder()
  {}

  void Recorder::record(unsigned int n_images)
  {
    size_t size = queue_.get_size();
    char* buffer = new char[size]();

    std::cout << "[RECORDER] started recording " << n_images << std::endl;

    for (unsigned int i = 0; i < n_images; ++i)
    {
      while (queue_.get_current_elts() < 1)
        std::this_thread::yield();

      queue_.dequeue(buffer);
      file_.write(buffer, size);
    }

    std::cout << "[RECORDER] recording has been stopped" << std::endl;

    delete[] buffer;
  }

  bool Recorder::is_file_exist(const std::string& filepath)
  {
    std::ifstream ifs(filepath);
    return ifs.good();
  }
}