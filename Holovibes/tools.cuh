/*! \file
 *
 * Various functions used most notably in the View panel. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

# ifndef _USE_MATH_DEFINES
/* Enables math constants. */
#  define _USE_MATH_DEFINES
# endif /* !_USE_MATH_DEFINES */
# include <math.h>

/* Forward declarations. */
namespace holovibes
{
  class Rectangle;
}
namespace holovibes
{
  struct UnwrappingResources;
}

/*! \brief  Apply a previously computed lens to image(s).
 *
 * The input data is multiplied element-wise with each corresponding
 * lens coefficient.
 *
 * \param input The input data to process in-place.
 * \param input_size Total number of elements to process. Should be a multiple
 * of lens_size.
 * \param lens The precomputed lens to apply.
 * \param lens_size The number of elements in the lens matrix.
 */
__global__ void kernel_apply_lens(
  cufftComplex *input,
  const unsigned int input_size,
  const cufftComplex *lens,
  const unsigned int lens_size);

// TODO 
__global__ void kernel_bursting(
	const cufftComplex *input,
	const unsigned int frame_resolution,
	const unsigned int nsamples,
	cufftComplex *output
	);
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

/*! \brief Shifts in-place the corners of an image.
 *
 * This function shifts zero-frequency components to the center
 * of the spectrum (and vice-versa), as explained in the matlab documentation
 * (http://fr.mathworks.com/help/matlab/ref/fftshift.html).
 *
 * \param input The image to modify in-place.
 * \param size_x The width of data, in pixels.
 * \param size_y The height of data, in pixels.
 * \param stream The CUDA stream on which to launch the operation.
 */
void shift_corners(
  float *input,
  const unsigned int size_x,
  const unsigned int size_y,
  cudaStream_t stream = 0);

/*! \brief Compute the log base-10 of every element of the input.
*
* \param input The image to modify in-place.
* \param size The number of elements to process.
* \param stream The CUDA stream on which to launch the operation.
*/
void apply_log10(
  float* input,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! \brief Apply the convolution operator to 2 complex matrices.
*
* \param First input matrix
* \param Second input matrix
* \param out Output matrix storing the result of the operation
* \param The number of elements in each matrix
* \param plan2d_x Externally prepared plan for x
* \param plan2d_k Externally prepared plan for k
* \param stream The CUDA stream on which to launch the operation.
*/
void convolution_operator(
  const cufftComplex* x,
  const cufftComplex* k,
  float* out,
  const unsigned int size,
  const cufftHandle plan2d_x,
  const cufftHandle plan2d_k,
  cudaStream_t stream = 0);

/*! \brief Extract a part of the input image to the output.
*
* \param input The full input image
* \param zone the part of the image we want to extract
* \param In pixels, the original width of the image
* \param Where to store the cropped image
* \param output_width In pixels, the desired width of the cropped image
* \param stream The CUDA stream on which to launch the operation.
*/
void frame_memcpy(
  float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width,
  cudaStream_t stream = 0);

/*! \brief Make the average of every element contained in the input.
 *
 * \param input The input data to average.
 * \param size The number of elements to process.
 * \param stream The CUDA stream on which to launch the operation.
 *
 * \return The average value of the *size* first elements.
 */
float average_operator(
  const float* input,
  const unsigned int size,
  cudaStream_t stream = 0);

/*! Let H be the latest complex image, and H-t the one preceding it.
 * This version computes : arg(H) - arg(H-t)
 * and unwraps the result.
 *
 * Phase unwrapping adjusts phase angles encoded in complex data,
 * by a cutoff value (which is here fixed to pi). Unwrapping seeks
 * two-by-two differences that exceed this cutoff value and performs
 * cumulative adjustments in order to 'smooth' the signal.
 */
void unwrap(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap);

/*! Let H be the latest complex image, H-t the conjugate matrix of
* the one preceding it, and .* the element-to-element matrix
* multiplication operation.
* This version computes : arg(H .* H-t)
* and unwraps the result.
*
* Phase unwrapping adjusts phase angles encoded in complex data,
* by a cutoff value (which is here fixed to pi). Unwrapping seeks
* two-by-two differences that exceed this cutoff value and performs
* cumulative adjustments in order to 'smooth' the signal.
*/
void unwrap_mult(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap);

/*! Let H be the latest complex image, and H-t the conjugate matrix of
* the one preceding it.
* This version computes : arg(H - H-t)
* and unwraps the result.
*
* Phase unwrapping adjusts phase angles encoded in complex data,
* by a cutoff value (which is here fixed to pi). Unwrapping seeks
* two-by-two differences that exceed this cutoff value and performs
* cumulative adjustments in order to 'smooth' the signal.
*/
void unwrap_diff(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap);
