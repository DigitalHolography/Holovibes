/*! \file
 *
 * \brief declaration of some useful unwrap function
 */
#pragma once

#include "common.cuh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"

/*! \brief Convert complex values to floating-point angles in [-pi; pi].
 *
 * Take complex data in cartesian form, and use conversion to polar
 * form to take the angle value of each element and store it
 * in a floating-point matrix. The resulting angles' values are bound
 * in [-pi; pi]. */
__global__ void kernel_extract_angle(float* output, const cuComplex* input, const size_t size);

/*! \brief Perform element-wise phase adjustment on a pixel matrix.
 *
 * \param output[out] Where to store the unwrapped version of cur.
 * \param pred[in] Predecessor phase image.
 * \param cur[in] Latest phase image.
 * \param size Size of an image in pixels. */
__global__ void kernel_unwrap(float* output, const float* pred, const float* cur, const size_t size);

/*! \brief Use the multiply-with-conjugate method to fill a float (angles) matrix.
 *
 * Computes cur .* conjugate(pred),
 * where .* is the element-wise complex-valued multiplication operation.
 * The angles of the resulting complex matrix are stored in output.
 * \param output[out] The matrix which shall store ther resulting angles.
 * \param pred[in] Predecessor complex image.
 * \param cur[in] Latest complex image.
 * \param size The size of an image in pixels. */
__global__ void
kernel_compute_angle_mult(float* output, const cuComplex* pred, const cuComplex* cur, const size_t size);

/*! \brief Use the subtraction method to fill a float (angles) matrix.
 *
 * Computes cur - conjugate(pred). The angles of the resulting complex matrix
 * are stored in output.
 * \param output[out] The matrix which shall store ther resulting angles.
 * \param pred[in] Predecessor complex image.
 * \param cur[in] Latest complex image.
 * \param size The size of an image in pixels. */
__global__ void
kernel_compute_angle_diff(float* output, const cuComplex* pred, const cuComplex* cur, const size_t size);

/*! \brief Iterate over saved phase corrections and apply them to an image.
 *
 * \param data[in out] The image to be corrected.
 * \param corrections[in] Pointer to the beginning of the phase corrections buffer.
 * \param image_size The number of pixels in a single image.
 * \param history_size The number of past phase corrections used. */
__global__ void
kernel_correct_angles(float* data, const float* corrections, const size_t image_size, const size_t history_size);

/*! \brief Initialise fx, fy, z matrix for unwrap 2d.
 *
 * \param input[in out] The matrix.
 * \param width Matrix width.
 * \param height height.
 * \param frame_res resolution.
 * \param fx buffer.
 * \param fy buffer.
 * \param z buffer.
 */
__global__ void kernel_init_unwrap_2d(
    const float* input, const uint width, const uint height, const uint frame_res, float* fx, float* fy, cuComplex* z);

/*! \brief  Multiply each pixels of a complex frame value by a float.
**	Done for 2 complexes.
*/
__global__ void kernel_multiply_complexes_by_floats_(
    cuComplex* output1, cuComplex* output2, const float* input1, const float* input2, const uint size);

/*! \brief  Multiply each pixels of two complexes frames value by a single
 * complex.
 */
__global__ void kernel_multiply_complexes_by_single_complex(cuComplex* output1,
                                                            cuComplex* output2,
                                                            const cuComplex input,
                                                            const uint size);

/*! \brief  Multiply each pixels of complex frames value by a single complex.
 */
__global__ void kernel_multiply_complex_by_single_complex(cuComplex* output, const cuComplex input, const uint size);

/*! \brief  Get conjugate complex frame. */
__global__ void kernel_conjugate_complex(cuComplex* output, const uint size);

/*! \brief  Multiply a complex frames by a complex frame.
 */
__global__ void kernel_multiply_complex_frames_by_complex_frame(cuComplex* output1,
                                                                cuComplex* output2,
                                                                const cuComplex* input,
                                                                const uint size);

/*! \brief  Multiply a complex frames by ratio from fx or fy and norm of fx and
 * fy.
 */
__global__ void
kernel_norm_ratio(cuComplex* output1, cuComplex* output2, const float* input1, const float* input2, const uint size);

/*! \brief  Add two complex frames into one.
 */
__global__ void kernel_add_complex_frames(cuComplex* output, const cuComplex* input, const uint size);

/*! \brief  Calculate phi for a frame.
 */
__global__ void kernel_unwrap2d_last_step(float* output, const cuComplex* input, const uint size);

/*! \brief Compute arg(H(t) .* H^*(t- T))
 *
 * Let H be the latest complex image, H-t the conjugate matrix of
 * the one preceding it, and .* the element-to-element matrix
 * multiplication operation.
 * This version computes : arg(H(t) .* H^*(t- T))
 *
 * Phase increase adjusts phase angles encoded in complex data,
 * by a cutoff value (which is here fixed to pi). Unwrapping seeks
 * two-by-two differences that exceed this cutoff value and performs
 * cumulative adjustments in order to 'smooth' the signal.
 */
void phase_increase(const cuComplex* cur,
                    holovibes::UnwrappingResources* resources,
                    const size_t image_size,
                    const cudaStream_t stream);

/*! \brief Main function for unwrap_2d calculations */
void unwrap_2d(float* output,
               float* input,
               const cufftHandle plan2d,
               holovibes::UnwrappingResources_2d* res,
               const camera::FrameDescriptor& fd,
               const cudaStream_t stream);

/*! \brief Gradient calculation for unwrap_2d calculations */
void gradient_unwrap_2d(const cufftHandle plan2d,
                        holovibes::UnwrappingResources_2d* res,
                        const camera::FrameDescriptor& fd,
                        const cudaStream_t stream);

/*! \brief Eq calculation for unwrap_2d calculations */
void eq_unwrap_2d(const cufftHandle plan2d,
                  holovibes::UnwrappingResources_2d* res,
                  const camera::FrameDescriptor& fd,
                  const cudaStream_t stream);

/*! \brief Phi calculation for unwrap_2d calculations */
void phi_unwrap_2d(float* output,
                   const cufftHandle plan2d,
                   holovibes::UnwrappingResources_2d* res,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t stream);
