#ifndef TOOLS_CUH
# define TOOLS_CUH

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
__global__ void kernel_bursting_roi(
  cufftComplex *input,
  unsigned int tl_x,
  unsigned int tl_y,
  unsigned int br_x,
  unsigned int br_y,
  unsigned int curr_elt,
  unsigned int nsamples,
  unsigned int width,
  cufftComplex *output);
__global__ void kernel_reconstruct_roi(
  cufftComplex* input,
  cufftComplex* output,
  unsigned int  input_width,
  unsigned int  input_height,
  unsigned int  output_width,
  unsigned int  p,
  unsigned int  nsample);
// TODO: Explain what this does.
__global__ void kernel_complex_divide(
  cufftComplex* image,
  unsigned int size,
  float divider);
__global__ void kernel_float_divide(
  float* input,
  unsigned int size,
  float divider);

/*! \brief  Permits to shift the corners of an image.
*
* This function shift zero-frequency component to center of spectrum
* as explaines in the matlab documentation(http://fr.mathworks.com/help/matlab/ref/fftshift.html).
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
* The 2 images should have the same size.
* The result value is given is out.
* The 2 used planes should be externally prepared (for performance reasons).
* For further informations: Autofocus of holograms based on image sharpness.
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
* The exracted aera should be less Than the input image.
* The result extracted image given is contained in output, the output should be preallocated.
* Coordonates of the extracted area are specified into the zone.
*/
void frame_memcpy(
  const float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width);

/*! \brief  Make the average of all pixels contained into the input image
* The size parameter is the number of pixels of the input image
*/
float average_operator(
  const float* input,
  const unsigned int size);

#endif /* !TOOLS_CUH */