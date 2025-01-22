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

const int MIN_RGB = 0;
const int MAX_RGB = 255;
const int BMP_IDENTIFICATOR_SIZE = 2;

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
  private:
    unsigned int width;
    unsigned int height;

    std::vector<float> cache_image_;

  private:
    /*! \brief Read a BMP file and store it in cache_image_.
     *
     * The image will be normalised in greyscale as a char buffer.
     *
     * \param[in] path the path of the BMP file
     * \return int 0 if the file was read successfully, -1 otherwise
     */
    int read_bmp(const char* path);

    /*! \brief Interpolate the greyscaled image to the frame descriptor size.
     *
     * \param[in] fd_width the width of the frame descriptor
     * \param[in] fd_height the height of the frame descriptor
     */
    void interpolate_filter(size_t fd_width, size_t fd_height);

  public:
    /*! \brief Read a BMP file interpolate it to the frame descriptor size and store it in cache_image_.
     *
     * \param[in] path the path of the BMP file
     * \param[in] fd_width the width of the frame descriptor
     * \param[in] fd_height the height of the frame descriptor
     */
    InputFilter(std::string path, size_t fd_width, size_t fd_height)
        : cache_image_()
    {
        if (read_bmp(path.c_str()) != -1)
            interpolate_filter(fd_width, fd_height);
    }

    InputFilter(InputFilter& InputFilter) = default;

    ~InputFilter() = default;

    /*! \brief Return the interpolated greyscaled image.
     *
     * \return std::vector<float> the interpolated greyscaled image
     */
    inline std::vector<float> get_input_filter() const { return cache_image_; }
};
} // namespace holovibes
