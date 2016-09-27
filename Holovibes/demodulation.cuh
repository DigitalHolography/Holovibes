/*! \file
*
* Functions that will be used to compute the stft.
*/
#pragma once


#include <cuda_runtime.h>
#include <cufft.h>

/*! \brief Allows demodulation in real time. Considering that we need (exactly like
*   an STFT) put every pixel in a particular order to apply an FFT and then reconstruct
*   the frame, please consider that this computation is very costly.
 
* !!! An explanation of how the computation is given in stft.cuh !!!

* \param input input buffer is where frames are taken for computation
* \param stft_buf the buffer which will be exploded
* \param stft_dup_buf the buffer that will receive the plan1d transforms
* \parem frame_resolution number of pixels in one frame.
* \param nsamples number of frames that will be used.

*/
void demodulation(
	cufftComplex* input,
	cufftComplex*                   stft_buf,
	cufftComplex*                   stft_dup_buf,
	const cufftHandle plan,
	const unsigned int frame_resolution,
	const unsigned int nsamples,
	cudaStream_t stream = 0);