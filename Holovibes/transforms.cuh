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

/*! \brief Compute a lens to apply to an image used by the fft1
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on which the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_quadratic_lens(	complex							*output,
										const camera::FrameDescriptor	fd,
										const float						lambda,
										const float						dist);

/*! \brief Compute a lens to apply to an image used by the fft2
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on wich the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_spectral_lens(	complex							*output,
										const camera::FrameDescriptor	fd,
										const float						lambda,
										const float						distance);