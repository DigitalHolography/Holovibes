/*! \file bw_area.cuh
 *
 * \brief function use for black and white area filter and open algo
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "common.cuh"

using uint = unsigned int;

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