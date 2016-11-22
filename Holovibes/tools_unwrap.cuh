/*! \file
 *
 * Kernels used by the unwrap* functions. */
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>

/*! \brief Convert complex values to floating-point angles in [-pi; pi].
 *
 * Take complex data in cartesian form, and use conversion to polar
 * form to take the angle value of each element and store it
 * in a floating-point matrix. The resulting angles' values are bound
 * in [-pi; pi]. */
__global__ void kernel_extract_angle(
  const cufftComplex* input,
  float* output,
  const size_t size);

/*! Perform element-wise phase adjustment on a pixel matrix.
 *
 * \param pred Predecessor phase image.
 * \param cur Latest phase image.
 * \param output Where to store the unwrapped version of cur.
 * \param size Size of an image in pixels. */
__global__ void kernel_unwrap(
  float* pred,
  float* cur,
  float* output,
  const size_t size);

/*! Use the multiply-with-conjugate method to fill a float (angles) matrix.
 *
 * Computes cur .* conjugate(pred),
 * where .* is the element-wise complex-valued multiplication operation.
 * The angles of the resulting complex matrix are stored in output.
 * \param pred Predecessor complex image.
 * \param cur Latest complex image.
 * \param output The matrix which shall store ther resulting angles.
 * \param size The size of an image in pixels. */
__global__ void kernel_compute_angle_mult(
  const cufftComplex* pred,
  const cufftComplex* cur,
  float* output,
  const size_t size);

/*! Use the subtraction method to fill a float (angles) matrix.
*
* Computes cur - conjugate(pred). The angles of the resulting complex matrix
* are stored in output.
* \param pred Predecessor complex image.
* \param cur Latest complex image.
* \param output The matrix which shall store ther resulting angles.
* \param size The size of an image in pixels. */
__global__ void kernel_compute_angle_diff(
  const cufftComplex* pred,
  const cufftComplex* cur,
  float* output,
  const size_t size);

/*! Iterate over saved phase corrections and apply them to an image.
*
* \param data The image to be corrected.
* \param corrections Pointer to the beginning of the phase corrections buffer.
* \param image_size The number of pixels in a single image.
* \param history_size The number of past phase corrections used. */
__global__ void kernel_correct_angles(
  float* data,
  const float* corrections,
  const size_t image_size,
  const size_t history_size);

/*! Initialise fx, fy, z matrix for unwrap 2d.
*
* \param Matrix width.
* \param Matrix height.
* \param Matrix resolution.
* \param 1TF or 2TF result.
* \param fx buffer.
* \param fy buffer.
* \param z buffer. */

__global__ void kernel_init_unwrap_2d(
	unsigned int width,
	unsigned int height,
	unsigned int frame_res,
	cufftComplex *input,
	float *fx,
	float *fy,
	cufftComplex *z);