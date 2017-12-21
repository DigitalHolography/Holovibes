/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Image format stored as a structure. */
#pragma once

namespace camera
{
	using	Endianness =
	enum
	{
		LittleEndian,
		BigEndian
	};

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

		unsigned short		width;		//!< Width of the frame in pixels.
		unsigned short		height;		//!< Height of the frame in pixels.
		unsigned int		depth;		//!< Byte depth during acquisition.
		Endianness			byteEndian;	//!< To each camera software its endianness. Useful for 16-bit cameras.
	};
}