/*! \file
 *
 * \brief function for using barycentre for mask
 */
#pragma once

#include <cuda_runtime.h>

/*!
 * \brief Compute the mean of the buffer of frames A on the dimensions 1 and 2 multiplied by the single frame B, the
 * only remaining dimension will be the third one e.g. the number of frames in A
 * Example: A is of dimensions 512x512x506 (e.g. 506 frames of size 512x512) and B is of dimensions 512x512x1 (e.g. a
 * single frame of 512x512)
 * The output will be of dimensions 1x1x506, each element i along the third axis (the only one remainng) is the mean of
 * all pixels of frames number i of A multiplied by its corresponding pixel in B.
 *
 * Parallelization is done frame per frame, indeed the function will call its internal kernel once for each frame
 *
 * This function was originally made for use inside the CircularVideoBuffer class
 *
 * \param[out] output The output buffer
 * \param[in] A Input buffer containing depth frames
 * \param[in] B Input buffer containing a single frame
 * \param[in] size  The size of a single frame
 * \param[in] depth The number of frames contained in A
 * \param[in] stream The CUDA stream to use
 */
void compute_multiplication_mean(float* output, float* A, float* B, size_t size, size_t depth, cudaStream_t stream);

/*!
 * \brief
 *
 * \param output
 * \param crv_circle_mask
 * \param input
 * \param width
 * \param height
 * \param barycentre_factor
 * \param stream
 * \param CRV_index
 *
 * \return int
 */
int compute_barycentre_circle_mask(float* output,
                                   float* crv_circle_mask,
                                   float* input,
                                   size_t width,
                                   size_t height,
                                   float barycentre_factor,
                                   cudaStream_t stream,
                                   int CRV_index = -1);