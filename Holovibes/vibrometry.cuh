/*! \file
 *
 * Hologram division (vibrometry) function. */
#pragma once

# include "cuda_shared.cuh"

/*! \brief For each pixel (P and Q) of the two images, this function
* will output on output (O) : \n
* Ox = (PxQx + PyQy) / (QxQx + QyQy) \n
* Oy = (PyQx - PxQy) / (QxQx + QyQy) \n
*
* \param frame_p the numerator image
* \param frame_q the denominator image
*/
void frame_ratio(	const complex	*frame_p,
					const complex	*frame_q,
					complex			*output,
					const uint		size,
					cudaStream_t	stream = 0);