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

/*! \brief This function allows us to apply a flowgraphy algorithm to a given amount
*   of frames. For every pixel of a frame, we take a given number of pixels around in 3 dimension
*   (time included) and then apply the formula given in Laser Speckle Flowgraphy optical
*   review of Naoki Konishi published in May 8, 2002.

* 
*
* \param input Buffer on which the convolution will be applied
* \param gpu_special_queue Queue that stores n frames to allow the time dimension computation
* \param gpu_special_queue_start_index Start index of the special queue.
* \param frame_resolution number of pixel of a frame
* \param frame_width Width of one frame
* \param nframes Number of frames 
* 
*/
void convolution_flowgraphy(cuComplex	*input,
						cuComplex		*gpu_special_queue,
						uint&			gpu_special_queue_start_index,
						const uint		gpu_special_queue_max_index,
						const uint		frame_resolution,
						const uint		frame_width,
						const uint		nframes,
						cudaStream_t	stream = 0);
