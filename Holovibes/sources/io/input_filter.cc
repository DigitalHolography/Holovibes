#include "input_filter.hh"

#include <algorithm>
#include <fstream>
#include <iostream>

#define RETURN_ERROR(...)                                                                                              \
    {                                                                                                                  \
        LOG_ERROR(__VA_ARGS__);                                                                                        \
        fclose(f);                                                                                                     \
        cache_image_.clear();                                                                                          \
        return -1;                                                                                                     \
    }

namespace holovibes
{

int InputFilter::read_bmp(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL)
        RETURN_ERROR("InputFilter::read_bmp: IO error could not find file");

    // Clear data if already holds information
    cache_image_.clear();
    bmp_identificator identificator;
    int e = static_cast<int>(fread(identificator.identificator, sizeof(identificator), 1, f));
    if (e < 0)
        RETURN_ERROR("InputFilter::read_bmp: IO error file too short (identificator)");

    // Check to make sure that the first two bytes of the file are the "BM"
    // identifier that identifies a bitmap image.
    if (identificator.identificator[0] != 'B' || identificator.identificator[1] != 'M')
        RETURN_ERROR("{} is not in proper BMP format.\n", path);

    bmp_header header;
    e = static_cast<int>(fread((char*)(&header), sizeof(header), 1, f));
    if (e < 0)
        RETURN_ERROR("InputFilter::read_bmp: IO error file too short (header)");

    bmp_device_independant_info di_info;
    e = static_cast<int>(fread((char*)(&di_info), sizeof(di_info), 1, f));
    if (e < 0)
        RETURN_ERROR("InputFilter::read_bmp: IO error file too short (di_info)");

    // Check for this here and so that we know later whether we need to insert
    // each row at the bottom or top of the image.
    if (di_info.height < 0)
        di_info.height = -di_info.height;

    // Extract image height and width from header
    this->width = di_info.width;
    this->height = di_info.height;

    // Reallocate the vector with the new size
    cache_image_.resize(width * height);

    // Only support for 24-bit images
    if (di_info.bits_per_pixel != 24)
        RETURN_ERROR("InputFilter::read_bmp: IO error invalid file ({} uses {}bits per pixel (bit depth). Bitmap only "
                     "supports 24bit.)",
                     path,
                     std::to_string(di_info.bits_per_pixel));

    // No support for compressed images
    if (di_info.compression != 0)
        RETURN_ERROR(
            "InputFilter::read_bmp: IO error invalid file ({} is compressed. Bitmap only supports uncompressed "
            "images.)",
            path);

    // Skip to bytecode
    e = fseek(f, header.bmp_offset, 0);

    // Read the rest of the data pixel by pixel
    unsigned char* pixel = new unsigned char[3];
    float color;

    // Read the pixels for each row and column of Pixels in the image.
    for (int row = 0; std::cmp_less(row, height); row++)
    {
        for (int col = 0; std::cmp_less(col, width); col++)
        {
            int index = row * width + col;

            // Read 3 bytes (b, g and r)
            e = static_cast<int>(fread(pixel, sizeof(unsigned char), 3, f));
            if (e < 0)
                RETURN_ERROR("InputFilter::read_bmp: IO error file too short (pixels)");

            // Convert to shade of grey with magic numbers (channel-dependant luminance perception)
            color = pixel[0] * 0.0722f + pixel[1] * 0.7152f + pixel[2] * 0.2126f;
            // Flatten in [0,1]
            color /= 255.0f;

            cache_image_.at(index) = color;
        }
        // Rows are padded so that they're always a multiple of 4
        // bytes. This line skips the padding at the end of each row.
        e = fseek(f, width % 4, std::ios::cur);
    }

    fclose(f);
    return 0;
}

void bilinear_interpolation(
    float* filter_input, float* filter_output, size_t width, size_t height, size_t fd_width, size_t fd_height)
{
    float w_ratio = (float)width / (float)fd_width;
    float h_ratio = (float)height / (float)fd_height;

    for (size_t y = 0; y < fd_height; y++)
    {
        for (size_t x = 0; x < fd_width; x++)
        {
            // Get the corresponding input float coordinates
            float input_x_float = (float)x * w_ratio;
            float input_y_float = (float)y * h_ratio;

            // Get the coordinates of the 4 neighboring pixels
            size_t floor_x = static_cast<size_t>(floorf(input_x_float));
            size_t ceil_x = std::min(width - 1, static_cast<size_t>(ceilf(input_x_float)));
            size_t floor_y = static_cast<size_t>(floorf(input_y_float));
            size_t ceil_y = std::min(height - 1, static_cast<size_t>(ceilf(input_y_float)));

            // Get the values of the 4 neighboring pixels
            float v00 = filter_input[floor_y * width + floor_x];
            float v01 = filter_input[floor_y * width + ceil_x];
            float v10 = filter_input[ceil_y * width + floor_x];
            float v11 = filter_input[ceil_y * width + ceil_x];

            // Compute the value of the output pixel using bilinear interpolation
            float q = 0.0f;
            if (ceil_x == floor_x)
            {
                // Very special case where all the points coincide
                if (ceil_y == floor_y)
                    q = v00;
                else // swap v00 with v01 and swap v10 with v11 as floor_x and ceil_x coincide
                    q = v00 * (ceil_y - input_y_float) + v10 * (input_y_float - floor_y);
            }
            else
            {
                // swap v00 with v10 and swap v01 with v11 as floor_y and ceil_y coincide
                if (ceil_y == floor_y)
                    q = v00 * (ceil_x - input_x_float) + v01 * (input_x_float - floor_x);
                else
                {
                    // General case
                    float q1 = v00 * (ceil_x - input_x_float) + v01 * (input_x_float - floor_x);
                    float q2 = v10 * (ceil_x - input_x_float) + v11 * (input_x_float - floor_x);
                    q = q1 * (ceil_y - input_y_float) + q2 * (input_y_float - floor_y);
                }
            }

            // Set the output pixel value
            filter_output[y * fd_width + x] = q;
        }
    }
}

void InputFilter::interpolate_filter(size_t fd_width, size_t fd_height)
{
    std::vector<float> copy_filter(cache_image_.begin(), cache_image_.end());

    cache_image_.resize(fd_width * fd_height);
    std::fill(cache_image_.begin(), cache_image_.end(), 0.0f);

    bilinear_interpolation(copy_filter.data(), cache_image_.data(), width, height, fd_width, fd_height);

    width = static_cast<unsigned int>(fd_width);
    height = static_cast<unsigned int>(fd_height);
}

} // namespace holovibes
