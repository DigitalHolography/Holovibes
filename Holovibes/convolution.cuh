
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>


void convolution_kernel(
	cufftComplex* input,
	cufftComplex* tmp_input,
	const unsigned int frame_resolution,
	const unsigned int frame_width,
	const unsigned int nframes,
	const cufftComplex* kernel,
	const unsigned int k_width,
	const unsigned int k_heigth,
	const unsigned int k_z,
	cudaStream_t stream);