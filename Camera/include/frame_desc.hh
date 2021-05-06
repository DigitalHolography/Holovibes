/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Image format stored as a structure. */
#pragma once

namespace camera
{
using Endianness = enum { LittleEndian = 0, BigEndian = 1 };

/*! This structure contains everything related to the format of the images
 * captured by the current camera.
 * Changing the camera used changes the frame descriptor, which will be used
 * in the rendering window and the holograms computations. */
struct FrameDescriptor
{
    //! Obtain the total frame size in bytes.
    unsigned int frame_size() const { return width * height * depth; }
    //! \brief Return the frame resolution (number of pixels).
    unsigned int frame_res() const { return width * height; }

    unsigned short width;  //!< Width of the frame in pixels.
    unsigned short height; //!< Height of the frame in pixels.
    unsigned int depth;    //!< Byte depth during acquisition.
    Endianness byteEndian; //!< To each camera software its endianness. Useful
                           //!< for 16-bit cameras.
};
} // namespace camera