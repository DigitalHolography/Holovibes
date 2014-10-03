#ifndef FRAME_DESC_HH
# define FRAME_DESC_HH

# include "camera.hh"

namespace camera
{
  typedef enum endianness
  {
    BIG_ENDIAN,
    LITTLE_ENDIAN
  } e_endianness;

  /*! This structure contains everything related to the image format. */
  typedef struct frame_descriptor
  {
    /*! Width of the frame. */
    unsigned int         width;
    /*! Height of the frame. */
    unsigned int         height;
    /*! Bit depth during acquisition. */
    unsigned char        bit_depth;
    /*! Size of pixels in micrometers. */
    float                pixel_size;
    /*! Endianness of bytes. */
    e_endianness         endianness;

  public:
    /* Helper functions. */
    unsigned char get_byte_depth() const
    {
      return bit_depth / 8 + (bit_depth % 8 ? 1u : 0u);
    }
    unsigned int get_frame_size() const
    {
      const unsigned int depth = get_byte_depth();
      const unsigned int size = width * height * depth;

      return size;
    }
  } s_frame_desc;
}

#endif /* !FRAME_HH */