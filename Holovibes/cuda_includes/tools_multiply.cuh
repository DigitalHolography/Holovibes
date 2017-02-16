/*! \file
 *
 * Matrix multiplication functions on different types. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
# endif
# include <math.h>

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

/*! \brief  Multiply each pixels of a complex frame value by a float. Done for 2 complexes.
*/
__global__ void kernel_multiply_complexes_by_floats_(
	const float* input1,
	const float* input2,
	cufftComplex* output1,
	cufftComplex* output2,
	const unsigned int size);

	/*! \brief  Multiply each pixels of two complexes frames value by a single complex.
	*/
__global__ void kernel_multiply_complexes_by_single_complex(
	cufftComplex* output1,
	cufftComplex* output2,
	const cufftComplex input,
	const unsigned int size);

/*! \brief  Multiply each pixels of complex frames value by a single complex.
*/
__global__ void kernel_multiply_complex_by_single_complex(
	cufftComplex* output,
	const cufftComplex input,
	const unsigned int size);

/*! \brief  Get conjugate complex frame.
*/
__global__ void kernel_conjugate_complex(
	cufftComplex* output,
	const unsigned int size);

/*! \brief  Multiply a complex frames by a complex frame.
*/
__global__ void kernel_multiply_complex_frames_by_complex_frame(
	cufftComplex* output1,
	cufftComplex* output2,
	const cufftComplex* input,
	const unsigned int size);

	/*! \brief  Multiply a complex frames by ratio from fx or fy and norm of fx and fy.
	*/
__global__ void kernel_norm_ratio(
	const float* input1,
	const float* input2,
	cufftComplex* output1,
	cufftComplex* output2,
	const unsigned int size);

/*! \brief  Add two complex frames into one.
*/
__global__ void kernel_add_complex_frames(
	cufftComplex* output,
	const cufftComplex* input,
	const unsigned int size);

/*! \brief  Calculate phi for a frame.
*/
__global__ void kernel_phi(
	float* output,
	const cufftComplex* input,
	const cufftComplex coeff,
	const unsigned int size);

__global__ void kernel_convergence(
	cufftComplex* input1,
	cufftComplex* input2);