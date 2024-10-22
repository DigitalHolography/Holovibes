/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "common.cuh"

/*! \brief Compute a lens to apply to an image used by the Fresnel transform (fft1)
 *
 * \param output The lens computed by the function.
 * \param fd File descriptor of the images on which the lens will be applied.
 * \param lambda Laser dependent wave lenght
 * \param dist z choosen
 * \param pixel_size size of pixels of the input
 */
__global__ void kernel_quadratic_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float dist, const float pixel_size);

/*! \brief Compute a lens to apply to an image used by the Angular Spectrum (fft2)
 *
 * \param output The lens computed by the function.
 * \param fd File descriptor of the images on wich the lens will be applied.
 * \param lambda Laser dependent wave lenght
 * \param distance z choosen
 * \param pixel_size size of pixels of the input
 */
__global__ void kernel_spectral_lens(
    cuComplex* output, const uint lens_side_size, const float lambda, const float distance, const float pixel_size);

/*! \brief The CUDA Kernel applying the circular mask to the image. Using euclidian distance and circle formula
 *   (x^2 +  y^2 = r^2)
 *  This kernel requires the center and the radius required for the circle.
 *
 *  To call the kernel the threads must be in 2d, follow this example:
 *  uint threads_2d = get_max_threads_2d();
 *  dim3 lthreads(threads_2d, threads_2d);
 *  dim3 lblocks(1 + (width - 1) / threads_2d, 1 + (height - 1) / threads_2d);
 *
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