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
  const unsigned int input_size,
  const cufftComplex *lens,
  const unsigned int lens_size);

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
  const cufftComplex *input,
  const unsigned int tl_x,
  const unsigned int tl_y,
  const unsigned int br_x,
  const unsigned int br_y,
  const unsigned int curr_elt,
  const unsigned int nsamples,
  const unsigned int width,
  const unsigned int size,
  cufftComplex *output);

/*! \brief Reconstruct bursted pixel from input
* into output
* \param p which image we are on
* \param nsample total number of images
*/
__global__ void kernel_reconstruct_roi(
  const cufftComplex* input,
  cufftComplex*       output,
  const unsigned int  input_width,
  const unsigned int  input_height,
  const unsigned int  output_width,
  const unsigned int  reconstruct_width,
  const unsigned int  reconstruct_height,
  const unsigned int  p,
  const unsigned int  nsample);

/*! \brief  Permits to shift the corners of an image.
*
* This function shift zero-frequency component to center of spectrum
* as explained in the matlab documentation(http://fr.mathworks.com/help/matlab/ref/fftshift.html).
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void shift_corners(
  float *input,
  const unsigned int size_x,
  const unsigned int size_y);

/*! \brief  compute the log of all the pixels of input image(s).
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The value of pixels is replaced by their log10 value
* This function makes the Kernel call for the user in order to make the usage of the previous function easier.
*/
void apply_log10(
  float* input,
  const unsigned int size);

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
  const unsigned int size,
  const cufftHandle plan2d_x,
  const cufftHandle plan2d_k);

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