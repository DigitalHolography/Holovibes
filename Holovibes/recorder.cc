#include "stdafx.h"
#include "recorder.hh"

#include <exception>
#include <cassert>
#include <thread>

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
      throw std::exception("overwriting an existing file");
#endif /* Overwrite is useful while debugging. */

    file_.open(filepath, std::ios::binary | std::ios::trunc);
  }

  Recorder::~Recorder()
  {}

  void Recorder::record(unsigned int n_images)
  {
    size_t size = queue_.get_size();
    void* buffer = malloc(size);

    for (unsigned int i = 0; i < n_images; ++i)
    {
      while (queue_.get_current_elts() < 1)
        std::this_thread::yield();

      queue_.dequeue(buffer, cudaMemcpyDeviceToHost);
      file_.write((const char*)buffer, size);
    }

    free(buffer);
  }

  bool Recorder::is_file_exist(const std::string& filepath)
  {
    std::ifstream ifs(filepath);
    return ifs.good();
  }
}