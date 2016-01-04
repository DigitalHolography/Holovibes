/*! \file */
#pragma once

# include <device_launch_parameters.h>
# include <cufft.h>
# include <frame_desc.hh>

/*! \brief Compute a lens to apply to an image used by the fft1
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on which the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_quadratic_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  const float lambda,
  const float dist);

/*! \brief Compute a lens to apply to an image used by the fft2
*
* \param output The lens computed by the function.
* \param fd File descriptor of the images on wich the lens will be applied.
* \param lambda Laser dependent wave lenght
* \param dist z choosen
*/
__global__ void kernel_spectral_lens(
  cufftComplex* output,
  const camera::FrameDescriptor fd,
  const float lambda,
  const float distance);

/*! Perform a matrix transpose on a rectangular or square matrix,
 * and store the result in an output matrix.
 * \param width Number of columns of the input matrix.
 * \param height Number of rows of the input matrix. */
template <typename Type>
__global__ void kernel_transpose(
  Type* input,
  Type* output,
  const unsigned width,
  const unsigned height)
{
  const unsigned index = blockDim.x * blockIdx.x * threadIdx.x;
  if (index >= width * height)
    return;

  const unsigned line = index / width;
  const unsigned column = index % width;

  output[column * height + line] = input[line * width + column];
}

/*! Change in-place the input matrix to its conjugate form.
 * The complex data contained in the matrix should be in cartesian form. */
__global__ void kernel_conjugate(
  cufftComplex* input,
  const size_t size);