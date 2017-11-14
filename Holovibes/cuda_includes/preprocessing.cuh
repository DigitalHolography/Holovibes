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

#pragma once

# include "Common.cuh"

/*! \brief Precompute the sqrt q sqrt vector of values in
* range 0 to n.
*
* \param n Number of values to compute.
* \param output Array of the sqrt values form 0 to n - 1,
* this array should have size greater or equal to n.
*/
void make_sqrt_vect(float			*out,
					const ushort	n,
					cudaStream_t	stream = 0);

/*! \brief Ensure the contiguity of images extracted from
 * the queue for any further processing.
 * This function also compute the sqrt value of each pixel of images.
 *
 * \param input the device queue from where images should be taken
 * to be processed.
 * \param output A bloc made of n contigus images requested
 * to the function.
 * \param n Number of images to ensure contiguity.
 * \param sqrt_array Array of the sqrt values form 0 to 65535
 * in case of 16 bit images or from 0 to 255 in case of
 * 8 bit images.
 *
 *
 *\note This function can be improved by specifying
 * img8_to_complex or img16_to_complex in the pipe to avoid
 * branch conditions. But it is no big deal.
 * Otherwise, the convert function are not called outside because
 * this function would need an unsigned short buffer that is unused
 * anywhere else.
 */
void make_contiguous_complex(holovibes::Queue&			input,
							cuComplex		*output,
							const uint		n,
							cudaStream_t	stream = 0);
