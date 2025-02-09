/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include <cuda_runtime.h>

/*!
 * \brief Computes the mean of the element-wise multiplication of two arrays for multiple depths.
 *
 * This function computes the mean of the element-wise multiplication of two arrays, `A` and `B`, for multiple depths.
 * It launches a CUDA kernel to perform the multiplication and reduction for each depth and then divides the results
 * by the size to compute the mean. The function uses the provided CUDA stream for asynchronous execution.
 *
 * \param [out] output Pointer to the output array where the mean values will be stored.
 * \param [in] A Pointer to the first input array.
 * \param [in] B Pointer to the second input array.
 * \param [in] size The number of elements in the input arrays.
 * \param [in] depth The number of depths to process.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 *
 * \note The function configures the kernel launch parameters based on the size of the input arrays
 *       and calls `cudaCheckError()` to check for any CUDA errors after the kernel launch. It also
 *       calls `map_divide` to divide the results by the size to compute the mean.
 */
void compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream);

/*!
 * \brief Computes a barycentre circle mask and applies it to an input array.
 *
 * This function computes a barycentre circle mask and applies it to an input array. It first finds the indices of the
 * maximum and minimum values in the input array. It then computes circle masks centered at these indices with a radius
 * determined by the barycentre factor. Finally, it applies the masks to the input array using a bitwise OR operation.
 *
 * \param [out] output Pointer to the output array where the result will be stored.
 * \param [out] crv_circle_mask Pointer to the array where the CRV circle mask will be stored.
 * \param [in] input Pointer to the input array.
 * \param [in] width The width of the input array.
 * \param [in] height The height of the input array.
 * \param [in] barycentre_factor The factor used to determine the radius of the circle masks.
 * \param [in] stream The CUDA stream to use for the kernel launch and memory operations.
 * \param [in,out] CRV_index The index of the CRV (minimum value) in the input array. If -1, the function will find the
 * minimum value index.
 *
 * \return The index of the CRV (minimum value) in the input array.
 *
 * \note The function uses Thrust to find the maximum and minimum value indices. It then computes the circle masks
 *       and applies them to the input array using the `apply_mask_or` function. The function assumes that the input
 *       array is a 2D array with dimensions `width` x `height`.
 */
int compute_barycentre_circle_mask(float* output,
                                   float* crv_circle_mask,
                                   float* input,
                                   size_t width,
                                   size_t height,
                                   float barycentre_factor,
                                   cudaStream_t stream,
                                   int CRV_index = -1);