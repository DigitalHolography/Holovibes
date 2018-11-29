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
#include "cufft_handle.hh"
using holovibes::cuda_tools::CufftHandle;

/*! \brief This function normalize the kernel

* \param gpu_kernel_buffer Buffer which is the kernel
* \param size Size of the frame (height * width)
*/
void normalize_kernel(float		*gpu_kernel_buffer_,
					  size_t	size);

/*! \brief This function allows us to apply a convolution (with a kernel) to frames

*
* \param input Buffer on which the convolution will be applied 
* \param convolved_buffer Buffer used for convolution calcul (will be overwriten)
* \param plan Plan2D used for the three fft
* \param frame_width Width of the frame
* \param frame_height Height of the frame
* \param kernel Array of float which is the convolution's kernel
* \param divide_convolution_enabled Activate the division of the input by the convolved image
*/
void convolution_kernel(float			*gpu_input,
						float			*gpu_convolved_buffer,
						CufftHandle		*plan,
						const uint		frame_width,
						const uint		frame_height,
						const float		*gpu_kernel,
						const bool		divide_convolution_enabled,
						const bool		normalize_enabled);
