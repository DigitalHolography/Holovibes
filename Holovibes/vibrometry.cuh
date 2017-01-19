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