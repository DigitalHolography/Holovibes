/*! \file
 *
 *  \brief Kernels functions to compute masks used in the pipe.
 *  The differents masks are :
 *  - Quadratic lens
 *  - Spectral lens
 *  - Cicular mask
 *  Check the functions to see the different behaviours.
 */
#pragma once

#include "common.cuh"
#include "cuda_memory.cuh"

/*! \brief Compute a lens to apply to an image used by the Fresnel transform (fft1)
 *
 * \param[out] output The lens computed by the function.
 * \param[in] fd File descriptor of the images on which the lens will be applied.
 * \param[in] lambda Laser dependent wave lenght
 * \param[in] dist z choosen
 * \param[in] pixel_size size of pixels of the input
 */
__global__ void kernel_quadratic_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float dist, const float pixel_size);

/*! \brief Compute a lens to apply to an image used by the Angular Spectrum (fft2)
 *
 * \param[out] output The lens computed by the function.
 * \param[in] fd File descriptor of the images on wich the lens will be applied.
 * \param[in] lambda Laser dependent wave lenght
 * \param[in] distance z choosen
 * \param[in] pixel_size size of pixels of the input
 */
__global__ void kernel_spectral_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float distance, const float pixel_size);

/*! \brief The CUDA Kernel getting a circular mask with a given center and radius.
 *  Using euclidian distance and circle formula (x^2 +  y^2 = r^2)
 *
 *  To call the kernel the threads must be in 2d, follow this example:
 *  ```cpp
 *  uint threads_2d = get_max_threads_2d();
 *  dim3 lthreads(threads_2d, threads_2d);
 *  dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);
 *  ```
 *
 *  \param[out] output The output image after mask application.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] center_X The x composite of the center of the image.
 *  \param[in] center_Y The y composite of the center of the image.
 *  \param[in] radius The radius of the circle.
 */
__global__ void
kernel_circular_mask(float* output, short width, short height, float center_X, float center_Y, float radius);

/*! \brief Get a circular mask with a given center and radius. Using euclidian distance and circle formula
 *   (x^2 +  y^2 = r^2)
 *
 *  \param[out] output The output image after mask application.
 *  \param[in] center_X The x composite of the center of the image.
 *  \param[in] center_Y The y composite of the center of the image.
 *  \param[in] radius The radius of the circle.
 *  \param[in] width The width of the image.
 *  \param[in] height The height of the image.
 *  \param[in] stream The stream CUDA to execute the kernel.
 */
void get_circular_mask(float* output,
                       const float center_X,
                       const float center_Y,
                       const float radius,
                       const short width,
                       const short height,
                       const cudaStream_t stream);