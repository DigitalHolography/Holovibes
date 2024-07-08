/*! \file
 *
 * \brief Contains all resources and functionalities used for 2D phase unwrapping.
 */
#pragma once

#include <cufft.h>

namespace holovibes
{
/*! \struct UnwrappingResources_2d
 *
 * \brief Manages the resources required for 2D phase unwrapping using CUDA.
 *
 * This struct encapsulates the necessary GPU resources, memory management,
 * and initialization required to perform 2D phase unwrapping operations.
 */
struct UnwrappingResources_2d
{
    /*! 
     * \brief Constructor to allocate necessary memory and initialize parameters.
     *
     * This constructor initializes the image resolution and CUDA stream, 
     * and allocates the necessary memory on the GPU for phase unwrapping operations.
     *
     * \param image_size The number of pixels in the image.
     * \param stream CUDA stream for asynchronous operations.
     */
    UnwrappingResources_2d(const size_t image_size, const cudaStream_t& stream);

    /*! 
     * \brief Destructor to deallocate any allocated GPU memory.
     *
     * Ensures that any allocated buffers are properly deallocated to avoid memory leaks.
     */
    ~UnwrappingResources_2d();

    /*! 
     * \brief Reallocates buffers based on the new image size.
     *
     * This function reallocates GPU buffers only if the current capacity
     * is insufficient for the new image size.
     *
     * \param ptr Pointer to the memory to be reallocated.
     * \param size The new size for the memory allocation.
     */
    void cudaRealloc(void* ptr, const size_t size);

    /*! 
     * \brief Reallocates all buffers based on the new image size.
     *
     * Ensures that all required buffers are reallocated if the current capacity
     * is insufficient for the new image size.
     *
     * \param image_size The number of pixels in the image.
     */
    void reallocate(const size_t image_size);

    /*! \brief The number of pixels in the image. */
    size_t image_resolution_;

    /*! \brief Buffer for the x-component of the frequency matrix. */
    float* gpu_fx_;
    /*! \brief Buffer for the y-component of the frequency matrix. */
    float* gpu_fy_;
    /*! \brief Buffer for the circularly shifted x-component of the frequency matrix. */
    float* gpu_shift_fx_;
    /*! \brief Buffer for the circularly shifted y-component of the frequency matrix. */
    float* gpu_shift_fy_;
    /*! \brief Buffer for storing the result of the 2D phase unwrapping. */
    float* gpu_angle_;
    /*! \brief Buffer for storing complex values in the frequency domain. */
    cufftComplex* gpu_z_;
    /*! \brief Common buffer for x-gradient and x-equation matrices. */
    cufftComplex* gpu_grad_eq_x_;
    /*! \brief Common buffer for y-gradient and y-equation matrices. */
    cufftComplex* gpu_grad_eq_y_;
    /*! \brief Buffer used for finding the minimum and maximum values. */
    float* minmax_buffer_;

    /*! \brief CUDA stream for asynchronous operations. */
    const cudaStream_t& stream_;
};
} // namespace holovibes