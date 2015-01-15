#include "transforms.cuh"

#include <device_launch_parameters.h>

#ifndef _USE_MATH_DEFINES
/* Enables math constants. */
# define _USE_MATH_DEFINES
#endif /* !_USE_MATH_DEFINES */
#include <math.h>

/*! \brief Compute a lens to apply to an image 
*
*
* \param n output The lens computed by the function.
* The output should have the same caracteristics of 
* of the images on wich the lens will be applied.
* \param fd File descriptor of the images on wich the lens will be applied.
*/
__global__ void kernel_quadratic_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float dist)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  float c = M_PI / (lambda * dist);
  float csquare;
  float dx = fd.pixel_size * 1.0e-6f;
  float dy = fd.pixel_size * 1.0e-6f;

  float x = (i - ((float)fd.width / 2)) * dx;
  float y = (j - ((float)fd.height / 2)) * dy;

  if (index < fd.width * fd.height)
  {
    csquare = c * (x * x + y * y);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}

/*! \brief Compute a lens to apply to an image
*
*
* \param n output The lens computed by the function.
* The output should have the same caracteristics of
* of the images on wich the lens will be applied.
* \param fd File descriptor of the images on wich the lens will be applied.
*/
__global__ void kernel_spectral_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  float lambda,
  float distance)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;

  float c = 2 * M_PI * distance / lambda;
  float csquare;

  float dx = fd.pixel_size * 1.0e-6f;
  float dy = fd.pixel_size * 1.0e-6f;

  float du = 1 / (((float)fd.width) * dx);
  float dv = 1 / (((float)fd.height) * dy);

  float u = (i - (float)(lrintf((float)fd.width / 2))) * du;
  float v = (j - (float)(lrintf((float)fd.height / 2))) * dv;

  if (index < fd.width * fd.height)
  {
    csquare = c * sqrtf(1.0f - lambda * lambda * u * u - lambda * lambda * v * v);
    output[index].x = cosf(csquare);
    output[index].y = sinf(csquare);
  }
}