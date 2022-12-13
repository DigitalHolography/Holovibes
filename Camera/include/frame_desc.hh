/*! \file
 *
 * Image format stored as a structure. */
#pragma once

#include <iostream>

namespace holovibes
{
using Endianness = enum { LittleEndian = 0, BigEndian = 1 };

/*! \brief This structure contains everything related to the format of the images captured by the current camera.
 *
 * Changing the camera used changes the frame descriptor, which will be used
 * in the rendering window and the holograms ComputeModeEnums.
 */
struct FrameDescriptor
{
    /*! \brief Obtain the total frame size in bytes. */
    size_t get_frame_size() const { return width * height * depth; }
    /*! \brief Return the frame resolution (number of pixels). */
    size_t get_frame_res() const { return width * height; }

    unsigned short width;  /*!< Width of the frame in pixels. */
    unsigned short height; /*!< Height of the frame in pixels. */
    unsigned int depth;    /*!< Byte depth during acquisition. */
    Endianness byteEndian; /*!< To each camera software its endianness. Useful for 16-bit cameras. */

    bool operator!=(const FrameDescriptor& rhs) const
    {
        return width != rhs.width || height != rhs.height || depth != rhs.depth;
    }
};

inline std::ostream& operator<<(std::ostream& os, const FrameDescriptor& fd)
{
    return os << "width : " << fd.width << ", height : " << fd.height << ", depth : " << fd.depth;
}
} // namespace holovibes
