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

/*! \brief  Apply a previously computed lens to image(s).
*
* The image(s) to treat, seen as input, should be contigous, the input_size is the total number of pixels to
* treat with the function.
*/
__global__ void kernel_apply_lens(
  cufftComplex *input,
  unsigned int input_size,
  cufftComplex *lens,
  unsigned int lens_size);

/*! \brief Will split the pixels of the original image
 * into output respecting the ROI selected
 * \param tl_x top left x coordinate of ROI
 * \param tl_y top left y coordinate of ROI
 * \param br_x bot right x coordinate of ROI
 * \param br_y bot right y coordinate of ROI
 * \param curr_elt which image out of nsamples we are doing the ROI on
 * \param width total width of input
 * \param output buffer containing all our pixels taken from the ROI
 */
__global__ void kernel_bursting_roi(
  cufftComplex *input,
  unsigned int tl_x,
  unsigned int tl_y,
  unsigned int br_x,
  unsigned int br_y,
  unsigned int curr_elt,
  unsigned int nsamples,
  unsigned int width,
  unsigned int size,
  cufftComplex *output);

/*! \brief Reconstruct bursted pixel from input
* into output
* \param p which image we are on
* \param nsample total number of images
*/
__global__ void kernel_reconstruct_roi(
  cufftComplex* input,
  cufftComplex* output,
  unsigned int  input_width,
  unsigned int  input_height,
  unsigned int  output_width,
  unsigned int  reconstruct_width,
  unsigned int  reconstruct_height,
  unsigned int  p,
  unsigned int  nsample);

/*! \brief Divide a complex image by a divider
* the two coordinate of each point (complex) will
* be divided
*/
__global__ void kernel_complex_divide(
  cufftComplex* image,
  unsigned int size,
  float divider);

/*! \brief Divide a float image by a divider
*/
__global__ void kernel_float_divide(
  float* input,
  unsigned int size,
  float divider);

/*! \brief  Permits to shift the corners of an image.
*
* This function shift zero-frequency component to center of spectrum
* as explained in the matlab documentation(http://fr.mathworks.com/help/matlab/ref/fftshift.html).
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void shift_corners(
  float *input,
  unsigned int size_x,
  unsigned int size_y);

/*! \brief  compute the log of all the pixels of input image(s).
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The value of pixels is replaced by their log10 value
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void apply_log10(
  float* input,
  unsigned int size);

/*! \brief  apply the convolution operator to 2 complex images (x,k).
*
* \param x first matrix
* \param k second matrix
* \param out output result
* \param plan2d_x externally prepared plan for x
* \param plan2d_k externally prepared plan for k
*/
void convolution_operator(
  const cufftComplex* x,
  const cufftComplex* k,
  float* out,
  unsigned int size,
  cufftHandle plan2d_x,
  cufftHandle plan2d_k);

/*! \brief  Extract a part of the input image to the output.
*
* \param zone the part of the image we want to extract
*/
void frame_memcpy(
  const float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width);

/*! \brief  Make the average of all pixels contained into the input image
*/
float average_operator(
  const float* input,
  const unsigned int size);

/*! \brief Perform a device-to-device memory copy from src to dst.
** \param nb_elts is the number of elements of type T to be copied.
** There is no need to take into account sizeof(T) in nb_elts.
*/
void copy_buffer(
  cufftComplex* src,
  cufftComplex* dst,
  const size_t nb_elts);