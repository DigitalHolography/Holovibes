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

    // Returns the pure normalised image in shades of grey as a char buffer AND sets the width and height of the object
    void read_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path);

    void interpolate_filter(
    float* filter_input, float* filter_output, size_t width, size_t height, size_t fd_width, size_t fd_height);

  public:
    InputFilter(std::shared_ptr<std::vector<float>> cache_image, std::string path){
      read_bmp(cache_image, path.c_str());
    }

    InputFilter(InputFilter& InputFilter) = default;

    ~InputFilter() = default;

    void apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream);
};
}