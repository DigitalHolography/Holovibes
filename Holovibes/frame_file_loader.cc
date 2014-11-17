#include "stdafx.h"
#include "frame_file_loader.hh"
#include <exception>

namespace holovibes
{
  FrameFileLoader::FrameFileLoader(
    const std::string& path,
    unsigned int nbframes,
    unsigned short width,
    unsigned short height,
    unsigned short depth,
    float pixel_size,
    camera::endianness endianness)
    : file_(path, std::ios::binary)
    , frame_desc_(
  {
    width,
    height,
    depth,
    pixel_size,
    endianness
  })
  , queue_(frame_desc_, nbframes)
  {
    if (!file_.is_open())
      throw std::runtime_error("[LOADER] unable to read/open file");

    const unsigned int frame_size = frame_desc_.frame_size();
    char* buffer = new char[frame_size];

    for (unsigned int i = 0; i < nbframes; ++i)
    {
      file_.read(buffer, frame_size);
      queue_.enqueue(buffer, cudaMemcpyHostToDevice);
    }

    delete[] buffer;
  }
}