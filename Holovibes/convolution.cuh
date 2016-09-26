
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

/*! \brief This function allows us to apply a convolution (with a kernel) to frames
            
* This algorithm is currently VERY ressource consuming and need to be improved.
*
* \param input is the buffer on which the convolution will be applied
* \param As the input buffer is going to be modified, we need a copy of it to 
*  apply convolution.
* \param resolution of one frame.
* \param width of one frame
* \param number of frame
* \param array of complex in which is stored the kernel
* \param kernel's width
* \param kernel's height
* \param kernel's depth
*/
void convolution_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
	const cufftComplex* kernel,
	const unsigned int k_width,
	const unsigned int k_height,
	const unsigned int k_z,
	cudaStream_t stream);