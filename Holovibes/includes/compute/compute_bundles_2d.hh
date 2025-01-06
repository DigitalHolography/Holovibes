/*! \file
 *
 * \brief Regroup all resources used for phase unwrapping 2d.
 */
#pragma once

#include <cufft.h>

namespace holovibes
{
/*! \struct UnwrappingResources_2d
 *
 * \brief Struct used for phase unwrapping 2d
 */
struct UnwrappingResources_2d
{
    /*! \brief Allocate with CUDA the required memory and initialize the image resolution and stream */
    UnwrappingResources_2d(const size_t image_size, const cudaStream_t& stream);

    /*! \brief If buffers were allocated, deallocate them. */
    ~UnwrappingResources_2d();

    /*! \brief Allocates all buffers based on the new image size.
     *
     * Reallocation is carried out only if the total amount of images
     * that can be stored in gpu_unwrap_buffer_ is inferior to
     * the capacity requested (in capacity_).
     *
     * \param image_size The number of pixels in an image.
     */
    void cudaRealloc(void* ptr, const size_t size);
    void reallocate(const size_t image_size);

    /*Image_size in pixel */
    size_t image_resolution_;

    /*! Matrix for fx */
    float* gpu_fx_;
    /*! Matrix for fy */
    float* gpu_fy_;
    /*! Matrix for cirshiffed fx */
    float* gpu_shift_fx_;
    /*! Matrix for cirshiffed fy */
    float* gpu_shift_fy_;
    /*! Matrix for unwrap_2d result */
    float* gpu_angle_;
    /*! Matrix for z */
    cufftComplex* gpu_z_;
    /*! Common matrix for grad_x and eq_x */
    cufftComplex* gpu_grad_eq_x_;
    /*! Common matrix for grad_y and eq_y */
    cufftComplex* gpu_grad_eq_y_;
    /*! Buffer to seek minmax value */
    float* minmax_buffer_;

    const cudaStream_t& stream_;
};
} // namespace holovibes
