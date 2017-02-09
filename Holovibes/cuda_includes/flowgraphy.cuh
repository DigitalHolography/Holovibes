#pragma once

# include "cuda_shared.cuh"

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
void convolution_flowgraphy(complex			*input,
							complex			*gpu_special_queue,
							uint			&gpu_special_queue_start_index,
							const uint		gpu_special_queue_max_index,
							const uint		frame_resolution,
							const uint		frame_width,
							const uint		nframes,
							cudaStream_t	stream);