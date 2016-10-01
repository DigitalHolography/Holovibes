#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>

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
void convolution_flowgraphy(
	cufftComplex* input,
	cufftComplex* gpu_special_queue,
	unsigned int &gpu_special_queue_start_index,
	const unsigned int gpu_special_queue_max_index,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
	cudaStream_t stream);