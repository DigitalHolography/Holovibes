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

/*! \brief This function allows us to apply a convolution (with a kernel) to frames
            
* This algorithm is currently VERY ressource consuming and need to be improved.
*
* \param input Buffer on which the convolution will be applied
* \param tmp_input As the input buffer is going to be modified, we need a copy of it to 
*  apply convolution.
* \param frame_resolution Resolution of one frame.
* \param frame_width Width of one frame
* \param nframes Number of frame
* \param kernel Array of complex which is the convolution's kernel
* \param k_width kernel's width
* \param k_height kernel's height
* \param k_z kernel's depth
*/
void convolution_kernel(cufftComplex	*input,
						cufftComplex	*gpu_special_queue,
						const uint		frame_resolution,
						const uint		frame_width,
						const float		*kernel,
						const uint		k_width,
						const uint		k_height,
						const uint		k_z,
						uint&			gpu_special_queue_start_index,
						const uint&		gpu_special_queue_max_index,
						cudaStream_t	stream);