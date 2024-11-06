/*! \file
 *
 * \brief function use for black and white area filter algo
 */
#pragma once

#include "frame_desc.hh"

/**
 * \brief write a matrix with connected component (labels_d) and store the size of each component in an other matrix
 * (label_sizes_d), this function use TwoPass algorithm, and does not handle the border of the image
 *
 * \param[in] image_d The image to process (GPU Memory)
 * \param[out] labels_d  The matrix use to store label of each pixel (GPU Memory)
 * \param[out] labels_sizes_d the matrix use to store size of each label (GPU Memory)
 * \param[out] linked_d the matrix use to store linked of each label to an other label (GPU Memory), no need to use it
 * after this function
 * \param[out] size_t_gpu_ Size_t on GPU Memory use to store a mutex
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void get_connected_component(const float* image_d,
                             size_t* labels_d,
                             size_t* labels_sizes_d,
                             size_t* linked_d,
                             size_t* size_t_gpu_,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream);

void get_n_max_index(size_t* labels_size_d, size_t nb_label, size_t* labels_max_d, size_t n, const cudaStream_t stream);

/**
 * \brief set to 0.0f pixels who is keep, 1.0f otherwise
 *
 * \param[in out] image_d The image to process (GPU Memory)
 * \param[in] label_d The matrix who store label of each pixel (GPU Memory)
 * \param[in] size Size of the frame
 * \param[in] is_keep_d Structure use to know if the label is keep in the output (GPU Memory)
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_filter(float* image_d, const size_t* label_d, size_t size, size_t* is_keep_d, const cudaStream_t stream);

/**
 * \brief transforme labels_sizes_d to is_keep_d
 *
 * \param[out] labels_sizes_d the matrix use to store value (GPU Memory)
 * \param[in] nb_labels Number of labels
 * \param[in] labels_max_d Tab of label whe want to keep (GPU Memory)
 * \param[in] n Size of labels_max_d
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void create_is_keep_in_label_size(
    size_t* labels_sizes_d, size_t nb_labels, size_t* labels_max_d, size_t n, const cudaStream_t stream);

/**
 * \brief return the number of label
 *
 * \param[in] labels_sizes_d Matrix who store each label size (GPU Memory)
 * \param[in] size Number of labels
 * \param[in out] size_t_gpu Size_t on GPU Memory use to store the result from cuda kernel
 * \param[in] stream The CUDA stream on which to launch the operation
 *
 * \return number of label
 */
int get_nb_label(size_t* labels_size_d, size_t size, size_t* size_t_gpu_, const cudaStream_t stream);
