/*! \file
 *
 * \brief Implementation of a circular queue
 *
 * Queue class is a custom circular FIFO data structure. It can handle
 * CPU or GPU data. This class is used to store the raw images, provided
 * by the camera, and holograms.
 */
#pragma once

#include "icompute.hh"

namespace holovibes
{
class InputFilter
{
    unsigned int width;
    unsigned int height;

    // Returns the pure image as a char buffer AND sets the width and height of the object
    unsigned char* read_bmp(char* path);

    void normalize_filter(const cudaStream_t stream);

    void interpolate_filter(size_t fd_width, size_t fd_height, const cudaStream_t stream);

  public:
    InputFilter(std::string path);

    InputFilter(InputFilter& InputFilter) = default;

    ~InputFilter() = default;

    void apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream);
};
}