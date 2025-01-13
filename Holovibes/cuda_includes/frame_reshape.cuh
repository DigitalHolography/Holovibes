/*! \file
 *
 * \brief Declaration of frame reshape functions
 */
#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

using uint = unsigned int;

/*! \brief Copies whole input image into output at position (output_srcx, output_srcy)
 *
 * \param input The full input image
 * \param input_width The input image's width in elements (number of elements on one row)
 * \param input_height The input image's height in elements (number of elements on one column)
 * \param output The full output image
 * \param output_width The output image's width in elements (number of elements on one row)
 * \param output_height The output image's height in elements (number of elements on one column)
 * \param output_startx The output image's subzone's top left corner's x coordinate
 * \param output_starty The output image's subzone's top left corner's y coordinate
 * \param elm_size The size of one element in bytes
 * \param kind The direction of the data transfer (host/device to host/device)
 * \param stream The cuda Stream
 */
cudaError_t embedded_frame_cpy(const char* input,
                               const uint input_width,
                               const uint input_height,
                               char* output,
                               const uint output_width,
                               const uint output_height,
                               const uint output_startx,
                               const uint output_starty,
                               const uint elm_size,
                               cudaMemcpyKind kind,
                               const cudaStream_t stream);

/*! \brief Copies whole input image into output such that the copy is centered
 * The output is therefore a square of side max(input_width, input_height)
 *
 * \param input The full input image
 * \param input_width The input image's width in elements (number of elements on one row)
 * \param input_height The input image's height in elements (number of elements on one column)
 * \param output The full output image (should be a square of side = max(input_width, input_height))
 * \param elm_size The size of one element in bytes
 * \param kind The direction of the data transfer (host/device to host/device)
 * \param stream The cuda Stream
 */
cudaError_t embed_into_square(const char* input,
                              const uint input_width,
                              const uint input_height,
                              char* output,
                              const uint elm_size,
                              cudaMemcpyKind kind,
                              const cudaStream_t stream);

/*! \brief Copies whole input image into output such that the copy is centered
 * The output is therefore a square of side max(input_width, input_height)
 *
 * \param input The full input image
 * \param input_width The input image's width in elements (number of elements on one row)
 * \param input_height The input image's height in elements (number of elements on one column)
 * \param output The full output image (should be a square of side = max(input_width, input_height))
 * \param batch_size Number of images in the batch
 * \param elm_size The size of one element in bytes
 * \param stream used for copy
 */
void batched_embed_into_square(const char* input,
                               const uint input_width,
                               const uint input_height,
                               char* output,
                               const uint batch_size,
                               const uint elm_size,
                               const cudaStream_t stream);

/*! \brief Crops input image into whole output image
 *
 * \param input The full input image
 * \param input_width The input image's width in elements (number of elements on one row)
 * \param input_height The input image's height in elements (number of elements on one column)
 * \param output The full output imag
 * \param crop_startx The input image's subzone's top left corner's x coordinate
 * \param crop_starty The input image's subzone's top left corner's y coordinate
 * \param crop_width The input image's subzone's width in elements (number of elements on one row)
 * \param crop_height The input image's subzone's height in elements (number of elements on one column)
 * \param elm_size The size of one element in bytes
 * \param kind The direction of the data transfer (host/device to host/device)
 * \param stream The cuda Stream
 */
cudaError_t crop_frame(const char* input,
                       const uint input_width,
                       const uint input_height,
                       const uint crop_start_x,
                       const uint crop_start_y,
                       const uint crop_width,
                       const uint crop_height,
                       char* output,
                       const uint elm_size,
                       cudaMemcpyKind kind,
                       const cudaStream_t stream);

/*! \brief Crops input (keeping the center and leaving the borders) as a square and copies the result into output
 * \param input The full image
 * \param input_width The full image's width in elements (number of elements in one row)
 * \param input_height The full image's height in element (number of elements in one column)
 * \param output The full output image (should be a square of size = min(input_width, input_height))
 * \param elm_size The size of one element in bytes
 * \param kind The direction of the data transfer (host/device to host/device)
 * \param stream The cuda Stream
 */
cudaError_t crop_into_square(const char* input,
                             const uint input_width,
                             const uint input_height,
                             char* output,
                             const uint elm_size,
                             cudaMemcpyKind kind,
                             const cudaStream_t stream);

/*! \brief Crops input (keeping the center and leaving the borders) as a square and copies the result into output
 * \param input The full image
 * \param input_width The full image's width in elements (number of elements in one row)
 * \param input_height The full image's height in element (number of elements in one column)
 * \param output The full output image (should be a square of size = min(input_width, input_height))
 * \param elm_size The size of one element in bytes
 * \param batch_size Number of images in the batch
 * \param stream The cuda Stream
 */
void batched_crop_into_square(const char* input,
                              const uint input_width,
                              const uint input_height,
                              char* output,
                              const uint elm_size,
                              const uint batch_size,
                              const cudaStream_t stream);

/*! \brief Subsamples the input image into a smaller output image
 * output_width == input_width / sample_step
 * output_height == input_height / sample_step
 * input_width % sample_step == 0
 * input_height % sample_step == 0
 *
 * input_width == input_height == 8 && sample_step == 2:
 * #...#...
 * ..#...#.
 * .#...#..
 * ...#...#
 * #...#...
 * ..#...#.
 * .#...#..
 * ...#...#
 *
 * \param input The input image
 * \param input_width The input image's width
 * \param input_height The input image's height
 * \param output The output image
 * \param sample_step The number of pixels to skip while subsampling
 * \param elm_size The size of one element (pixel) in bytes
 * \param stream The cuda Stream
 */
void subsample_frame(const char* input,
                     const uint input_width,
                     const uint input_height,
                     char* output,
                     const uint sample_step,
                     const uint elm_size,
                     const cudaStream_t stream);

void subsample_frame_complex_batched(const cuComplex* input,
                                     const uint input_width,
                                     const uint input_height,
                                     cuComplex* output,
                                     const uint sample_step,
                                     const uint batch_size,
                                     const cudaStream_t stream);
