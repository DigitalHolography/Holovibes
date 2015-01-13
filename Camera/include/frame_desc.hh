#ifndef FRAME_DESC_HH
# define FRAME_DESC_HH

namespace camera
{
  typedef enum endianness
  {
    BIG_ENDIAN,
    LITTLE_ENDIAN
  } e_endianness;

  /*! \brief This structure contains everything related to the image format. */
  struct FrameDescriptor
  {
    /*! Width of the frame. */
    unsigned short         width;
    /*! Height of the frame. */
    unsigned short         height;
    /*! Byte depth during acquisition. */
    unsigned short         depth;
    /*! Size of pixels in micrometers. */
    float                pixel_size;
    /*! Endianness of bytes. */
    e_endianness         endianness;

  public:
    /* Helper functions. */

    /*! \brief Return the frame size in bytes. */
    unsigned int frame_size() const
    {
      return width * height * depth;
    }

    /*! \brief Return the frame resolution (number of pixels). */
    unsigned int frame_res() const
    {
      return width * height;
    }
  };
}

#endif /* !FRAME_DESC_HH */
