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
  
const int MIN_RGB=0;
const int MAX_RGB=255;
const int BMP_IDENTIFICATOR_SIZE=2;

// Windows BMP-specific format data
struct bmp_identificator
{
    unsigned char identificator[BMP_IDENTIFICATOR_SIZE];
};

struct bmp_header
{
	unsigned int file_size;
	unsigned short creator1;
	unsigned short creator2;
	unsigned int bmp_offset;
};

struct bmp_device_independant_info
{
  unsigned int header_size;
  int width;
  int height;
  unsigned short num_planes;
  unsigned short bits_per_pixel;
  unsigned int compression;
  unsigned int bmp_byte_size;
  int hres;
  int vres;
  unsigned int num_colors;
  unsigned int num_important_colors;
};
class InputFilter
{
    unsigned int width;
    unsigned int height;

    // Returns the pure normalised image in shades of grey as a char buffer AND sets the width and height of the object
    void read_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path);

    // Only for debug purposes
    void write_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path);

    void interpolate_filter(std::shared_ptr<std::vector<float>> cache_image, size_t fd_width, size_t fd_height);

  public:
    InputFilter(std::shared_ptr<std::vector<float>> cache_image, std::string path, size_t fd_width, size_t fd_height){
      read_bmp(cache_image, path.c_str());
      //interpolate_filter(cache_image, fd_width, fd_height);
      write_bmp(cache_image, path.c_str());
    }

    InputFilter(InputFilter& InputFilter) = default;

    ~InputFilter() = default;
};
}