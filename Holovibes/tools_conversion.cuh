/*! \file */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

# include "geometry.hh"

/* CONVERSION FUNCTIONS */

/*! \brief  This function permit to transform an 8 bit image to her complexe representation.
*
* The transformation is performed by putting the squareroot of the pixels value into the real an imagiginary
* part of the complex output.
* The input image is seen as unsigned char(8 bits data container) because of her bit depth.
* The result is given in output.
*/
__global__ void img8_to_complex(
  cufftComplex* output,
  const unsigned char* input,
  const unsigned int size,
  const float* sqrt_array);

/*! \brief  This function permit to transform an 16 bit image to her complexe representation
*
* The transformation is performed by putting the squareroot of the pixels value into the real an imagiginary
* into the complex output.
* The input image is seen as unsigned short(16 bits data container) because of her bitbytebytedepth.
* The result is given in output.
*/
__global__ void img16_to_complex(
  cufftComplex* output,
  const unsigned short* input,
  const unsigned int size,
  const float* sqrt_array);

/*! \brief  Compute the modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief  Compute the squared modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_squared_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief  Compute the arguments of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_argument(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! Copy the angle value of each element of the input.
 *
 * The input image should contain complex data in polar form.
 * Hence it simply copies the angle off each element to put it
 * to the output.
 *
 * \param input Complex-valued (in polar form) data.
 */
void complex_to_angle(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief  Convert the endianness of input image(s) from big endian to little endian.
*
* The image(s) to treat, seen as input, should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief  Convert all the pixels of input image(s) into unsigned short datatype.
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void float_to_ushort(
  const float* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream = 0);