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

#include "Common.cuh"

/*! \brief Make the contrast of the image depending of the
* maximum and minimum input given by the user.
*
* The algortihm used is a contrast stretching, the
* values min and max can be found thanks to the previous functions
* or can be set by the user in case of a particular use.
* \param input The image in gpu to correct contrast.
* \param size Size of the image in number of pixels.
* \param dynamic_range Range of pixel values
* \param min Minimum pixel value of the input image.
* \param max Maximum pixel value of the input image.
*
*/
void apply_contrast_correction(float* const	input,
							   const uint	size,
							   const ushort	dynamic_range,
							   const float	min,
							   const float	max);
