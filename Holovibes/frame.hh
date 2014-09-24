#ifndef FRAME_HH
# define FRAME_HH

# include "endianness.hh"

namespace camera
{
  class Frame
  {
  public:
    Frame()
    {}

    Frame(
      void* data,
      unsigned int width,
      unsigned int height,
      unsigned char pixel_depth,
      e_endianness endianness)
      : data_(data),
      width_(width),
      height_(height),
      pixel_depth_(pixel_depth),
      endianness_(endianness)
    {
    }
    ~Frame();

    void* get_data() const
    {
      return data_;
    }

    unsigned int get_width() const
    {
      return width_;
    }

    unsigned int get_height() const
    {
      return height_;
    }

    unsigned char get_pixel_depth_() const
    {
      return pixel_depth_;
    }

    e_endianness get_endianness() const
    {
      return endianness_;
    }

    unsigned int get_size() const
    {
      return width_ * height_ * pixel_depth_;
    }

  private:
    void* data_;
    unsigned int width_;
    unsigned int height_;
    unsigned char pixel_depth_;
    e_endianness endianness_;
  };
}

#endif /* !FRAME_HH */