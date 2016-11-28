/*! \file
 *
 * Matrix division functions on different types. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

# include "compute_descriptor.hh"

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* \param image The image(s) to process. Should be contiguous memory.
* \param size Number of elements to process.
* \param divider Divider value for all elements.
*/
__global__ void kernel_complex_divide(
  cufftComplex* image,
  const unsigned int size,
  const float divider);

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* \param image The image(s) to process. Should be contiguous memory.
* \param size Number of elements to process.
* \param divider Divider value for all elements.
*/
__global__ void kernel_float_divide(
  float* input,
  const unsigned int size,
  const float divider);


/*! \brief  Multiply the pixels value of 2 complexe input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_complex(
	const cufftComplex* input1,
	const cufftComplex* input2,
	cufftComplex* output,
	const unsigned int size);

/*! \brief  Multiply the pixels value of 2 float input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_float(
	const float* input1,
	const float* input2,
	float* output,
	const unsigned int size);


//TODO:
__global__ void kernel_substract_ref(
	cufftComplex* input,
	void*         reference,
	const holovibes::ComputeDescriptor compute_desc,
	const unsigned int nframes);

void substract_ref(
	cufftComplex* input,
	cufftComplex* reference,
	const unsigned frame_resolution,
	const unsigned int nframes,
	cudaStream_t stream = 0);

void mean_images(
	cufftComplex* input,
	cufftComplex* output,
	unsigned int n,
	unsigned int frame_size,
	cudaStream_t stream = 0);