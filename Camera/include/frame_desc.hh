#pragma once

namespace camera
{
  typedef enum endianness
  {
    BIG_ENDIAN,
    LITTLE_ENDIAN
  } e_endianness;

  /*! This structure contains everything related to the format of the images
   * captured by the current camera.
   * Changing the camera used changes the frame descriptor, which will be used
   * in the rendering window and the holograms computations. */
  struct FrameDescriptor
  {
  public:
    //!< Obtain the total frame size in bytes.
    unsigned int frame_size() const
    {
      return width * height * depth;
    }

    //!< \brief Return the frame resolution (number of pixels).
    unsigned int frame_res() const
    {
      return width * height;
    }

  public:
    unsigned short width; //!< Width of the frame in pixels.

    unsigned short height; //!< Height of the frame in pixels.

    float depth; //!< Byte depth during acquisition.

    float pixel_size; //!< Size of pixels in micrometers.

    //!< To each camera software its endianness. Useful for 16-bit cameras.
    e_endianness endianness;
  };
}