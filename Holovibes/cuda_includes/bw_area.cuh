/*! \file
 *
 * \brief function use for black and white area filter and open algo
 */
#pragma once

#include "frame_desc.hh"

using uint = unsigned int;

/**
 * \brief write a matrix with connected component (labels_d) and store the size of each component in an other matrix
 * (label_sizes_d), this function use TwoPass algorithm, and does not handle the border of the image
 *
 * \param[out] labels_d  The matrix use to store label of each pixel (GPU Memory)
 * \param[out] labels_sizes_d the matrix use to store size of each label (GPU Memory)
 * \param[out] linked_d the matrix use to store linked of each label to an other label (GPU Memory), no need to use it
 * after this function
 * \param[in] image_d The image to process (GPU Memory)
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void get_connected_component(uint* labels_d,
                             float* labels_sizes_d,
                             uint* linked_d,
                             const float* image_d,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream);

/**
 * \brief Sets to 1.0f each pixel that we want to keep from the selected label. Otherwise the pixels are set to 0.0f.
 * Hence only pixels with the selected label are retained
 *
 * \param[in out] image_d The image to process (GPU Memory)
 * \param[in] label_d The matrix who store label of each pixel (GPU Memory)
 * \param[in] size Size of the frame
 * \param[in] label_to_keep Label to keep
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_filter(float* image_d, const uint* label_d, size_t size, uint label_to_keep, const cudaStream_t stream);

/*!
 * \brief  Sets to 1.0f each pixel that we want to keep from connected component who are bigger than p. Otherwise the
 * pixels are set to 0.0f. Hence only pixels with the selected label are retained
 *
 * \param[in out] image_d The image to process (GPU Memory)
 * \param[in] label_d The matrix who store label of each pixel (GPU Memory)
 * \param[in] labels_sizes_d The matrix who store the size of eche labeled connected component (GPU Memory)
 * \param[in] size Size of the frame
 * \param[in] p The threshold of the size we want to keep
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_open(
    float* image_d, const uint* label_d, const float* labels_sizes_d, size_t size, uint p, const cudaStream_t stream);