/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_desc.hh"

/* use TwoPasse algo

    labels is the output and must be set to 0 and with the size of the image

    this function does not handle the border of the image
*/
/**
 * \brief TODO
 *
 * \param[in out] input The image to process
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void get_connected_component(const float* image_d,
                             size_t* labels_d,
                             size_t* labels_sizes_d,
                             size_t* linked_d,
                             const size_t width,
                             const size_t height,
                             const cudaStream_t stream);

void get_n_max_index(size_t* labels_size_d, size_t nb_label, size_t* labels_max_d, size_t n, const cudaStream_t stream);

/**
 * \brief TODO
 *
 * \param[in out] image_d The image to process
 * \param[in] label_d TODO
 * \param[in] size Size of the frame
 * \param[in] is_keep_d TODO
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_filter(float* image_d, const size_t* label_d, size_t size, size_t* is_keep_d, const cudaStream_t stream);

void create_is_keep_in_label_size(
    size_t* labels_sizes_d, size_t nb_labels, size_t* labels_max_d, size_t n, const cudaStream_t stream);

int get_nb_label(size_t* labels_size_d, size_t size, const cudaStream_t stream);
