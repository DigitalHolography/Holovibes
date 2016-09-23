/*! \file
*
* Functions that will be used to compute the stft.
*/
#pragma once


#include <cuda_runtime.h>
#include <cufft.h>

/*! \brief Call an fft1 1D on the image
*
* \param plan the first paramater of cufftExecC2C that will be called
* on the image
*/
void demodulation(
	cufftComplex* input,
	cufftComplex*                   stft_buf,
	cufftComplex*                   stft_dup_buf,
	const cufftHandle plan,
	const unsigned int frame_resolution,
	const unsigned int nsamples,
	cudaStream_t stream = 0);