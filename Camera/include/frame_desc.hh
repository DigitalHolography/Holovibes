/*! \file
 *
 * Image format stored as a structure. */
#pragma once

namespace camera
{
using Endianness = enum { LittleEndian = 0, BigEndian = 1 };

/*! \brief This enum is used to clarify the 'depth' parameter of the following FrameDescriptor struct.
 *  Basically, it's the number of bytes (8 bits) of data that each pixel of a frame needs.
 */
enum class PixelDepth
{
    Bits8 = 1,
    Bits16 = 2,
    Bits24 = 3,
    Bits32 = 4,
    Bits48 = 6,
    Bits64 = 8, // For complex values
    Composite = 12, // Needed by ImgType::Composite as the depth is 3 * sizeof(float)
    Bits0 = 0,
};

/*! \brief This structure contains everything related to the format of the images captured by the current camera.
 *
 * Changing the camera used changes the frame descriptor, which will be used
 * in the rendering window and the holograms computations.
 */
struct FrameDescriptor
{
    /*! \brief Obtain the total frame size in bytes. */
    size_t get_frame_size() const { 
        size_t size = width * height * static_cast<int>(depth);
        return size; 
    }
    /*! \brief Return the frame resolution (number of pixels). */
    size_t get_frame_res() const { return width * height; }

    unsigned short width;  /*!< Width of the frame in pixels. */
    unsigned short height; /*!< Height of the frame in pixels. */
    PixelDepth depth;    /*!< Byte depth during acquisition. */
    Endianness byteEndian; /*!< To each camera software its endianness. Useful for 16-bit cameras. */
};
} // namespace camera
