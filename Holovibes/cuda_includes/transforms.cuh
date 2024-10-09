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
