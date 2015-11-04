#pragma once

# include <string>
# include <fstream>
# include "queue.hh"
# include <frame_desc.hh>

namespace holovibes
{
  class FrameFileLoader
  {
  public:
    FrameFileLoader(
      const std::string& path,
      unsigned int nbframes,
      unsigned short width,
      unsigned short height,
      unsigned short depth,
      float pixel_size,
      camera::endianness endianness);

    ~FrameFileLoader()
    {}

    Queue& get_queue()
    {
      return queue_;
    }

  private:
    std::ifstream file_;
    camera::FrameDescriptor frame_desc_;
    Queue queue_;
  };
}