/*! \file bw_area.cuh
 *
 * \brief function use for black and white area filter and open algo
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "common.cuh"

using uint = unsigned int;

/**
 * \brief write a matrix with connected component (labels_d) and store the size of each component in an other matrix
 * (label_sizes_d), this function use TwoPass algorithm, and does not handle the border of the image
 *
 * \param[out] labels_d  The matrix use to store label of each pixel (GPU Memory)
 * \param[in] image_d The image to process (GPU Memory)
 * \param[in] width Width of the frame
 * \param[in] height Height of the frame
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void get_connected_component(uint* labels_d,
                             uint* linked_d,
                             const float* image_d,
                             const size_t width,
                             const size_t height,
                             size_t* change_d,
                             const cudaStream_t stream);

/**
 * \brief Sets to 1.0f each pixel that we want to keep from the selected label. Otherwise the pixels are set to
 * 0.0f. Hence only pixels with the selected label are retained
 *
 * \param[in out] input_output The image to process (GPU Memory)
 * \param[in] label_d The matrix who store label of each pixel (GPU Memory)
 * \param[in] size Size of the frame
 * \param[in] label_to_keep Label to keep
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_filter(float* input_output, const uint* label_d, size_t size, uint label_to_keep, const cudaStream_t stream);

/*!
 * \brief  Sets to 1.0f each pixel that we want to keep from connected component who are bigger than p. Otherwise the
 * pixels are set to 0.0f. Hence only pixels with the selected label are retained
 *
 * \param[in out] input_output The image to process (GPU Memory)
 * \param[in] label_d The matrix who store label of each pixel (GPU Memory)
 * \param[in] labels_sizes_d The matrix who store the size of eche labeled connected component (GPU Memory)
 * \param[in] size Size of the frame
 * \param[in] threshold The threshold of the size we want to keep
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void area_open(float* input_output,
               const uint* label_d,
               const float* labels_sizes_d,
               size_t size,
               uint threshold,
               const cudaStream_t stream);

/*!
 * \brief compute black and white area filter, find the largest connected component from a binarised image and remove
 * all the other, for this we use the ccl (connected component labeling) union find Kuroma algorithm
 *
 * \param[in out] input_output image to process
 * \param[in] width width of the image
 * \param[in] height height of the image
 * \param[in] labels_d GPU buffer of uint, this size must be the width * height, use for keep label of pixel in memory
 * \param[in] linked_d GPU buffer of uint, this size must be the width * height, use for keep label link in memory
 * \param[in] labels_sizes_d GPU buffer of float, this size must be the width * height, use for keep labels sizes in
 * memory, float is for use cublas max searsh
 * \param[in] change_d GPU size_t use to inside of cll algo
 * \param[in] handle cublas singleton
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void bwareafilt(float* input_output,
                size_t width,
                size_t height,
                uint* labels_d,
                uint* linked_d,
                float* labels_sizes_d,
                size_t* change_d,
                cublasHandle_t& handle,
                cudaStream_t stream);

/*!
 * \brief compute black and white area open, find the size of each connected component and remove all connected
 * component smaller than n, for this we use the ccl (connected component labeling) union find Kuroma algorithm
 *
 * \param[in out] input_output image to process
 * \param[in] n threashold size for connected component we want to keep
 * \param[in] width width of the image
 * \param[in] height height of the image
 * \param[in] labels_d GPU buffer of uint, this size must be the width * height, use for keep label of pixel in memory
 * \param[in] linked_d GPU buffer of uint, this size must be the width * height, use for keep label link in memory
 * \param[in] labels_sizes_d GPU buffer of float, this size must be the width * height, use for keep labels sizes in
 * memory, float is for use cublas max searsh
 * \param[in] change_d GPU size_t use to inside of cll algo
 * \param[in] stream The CUDA stream on which to launch the operation
 */
void bwareaopen(float* input_output,
                uint n,
                size_t width,
                size_t height,
                uint* labels_d,
                uint* linked_d,
                float* labels_sizes_d,
                size_t* change_d,
                cudaStream_t stream);