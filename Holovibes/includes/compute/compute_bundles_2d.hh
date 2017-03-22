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
* Regroup all resources used for phase unwrapping 2d. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# include "config.hh"


namespace holovibes
{
	struct UnwrappingResources_2d
	{
		/*! Initialize the capacity from history_size,
		* set size and next_index to zero, and buffers pointers to null pointers. */
		UnwrappingResources_2d(const size_t image_size);

		/*! If buffers were allocated, deallocate them. */
		~UnwrappingResources_2d();

		/*! Allocates all buffers based on the new image size.
		*
		* Reallocation is carried out only if the total amount of images
		* that can be stored in gpu_unwrap_buffer_ is inferior to
		* the capacity requested (in capacity_).
		*
		* \param image_size The number of pixels in an image. */
		void reallocate(const size_t image_size);

		/*Image_size in pixel*/
		size_t image_resolution_;


		/*! Matrix for fx. */
		float* gpu_fx_;
		/*! Matrix for fy */
		float* gpu_fy_;
		/*! Matrix for cirshiffed fx. */
		float* gpu_shift_fx_;
		/*! Matrix for cirshiffed fy */
		float* gpu_shift_fy_;
		/*! Matrix for unwrap_2d result*/
		float* gpu_angle_;
		/*! Matrix for z */
		cufftComplex* gpu_z_;
		/*! Common matrix for grad_x and eq_x */
		cufftComplex* gpu_grad_eq_x_;
		/*! Common matrix for grad_y and eq_y */
		cufftComplex* gpu_grad_eq_y_;
		/*! Buffer to seek minmax value */
		float* minmax_buffer_;
	};
}