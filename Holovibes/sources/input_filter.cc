#include "input_filter.hh"

#include <iostream>
#include <fstream>

namespace holovibes
{
void InputFilter::read_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path)
{
    FILE* f = fopen(path, "rb");
    if(f == NULL)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error could not find file");
        exit(0);
    }
    // Clear data if already holds information
    cache_image->clear();
	bmp_identificator identificator;
    int e = fread(identificator.identificator, sizeof(identificator), 1, f);
    if(e < 0)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error file too short (identificator)");
        exit(0);
    }

	// Check to make sure that the first two bytes of the file are the "BM"
	// identifier that identifies a bitmap image.
	if (identificator.identificator[0] != 'B' || identificator.identificator[1] != 'M')
	{
		LOG_ERROR("{} is not in proper BMP format.\n", path);
        exit(0);
	}

    bmp_header header;
    e = fread((char*)(&header), sizeof(header), 1, f);
    if(e < 0)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error file too short (header)");
        exit(0);
    }

	bmp_device_independant_info di_info;
    e = fread((char*)(&di_info), sizeof(di_info), 1, f);
    if(e < 0)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error file too short (di_info)");
        exit(0);
    }

	// Check for this here and so that we know later whether we need to insert
	// each row at the bottom or top of the image.
	if (di_info.height < 0)
	{
		di_info.height = -di_info.height;
	}

    // Extract image height and width from header
    this->width = di_info.width;
    this->height = di_info.height;

    // Reallocate the vector with the new size
    cache_image->resize(width * height);

	// Only support for 24-bit images
	if (di_info.bits_per_pixel != 24)
	{
        LOG_ERROR("InputFilter::read_bmp: IO error invalid file ({} uses {}bits per pixel (bit depth). Bitmap only supports 24bit.)", path, std::to_string(di_info.bits_per_pixel));
        exit(0);
	}

	// No support for compressed images
	if (di_info.compression != 0)
	{
        LOG_ERROR("InputFilter::read_bmp: IO error invalid file ({} is compressed. Bitmap only supports uncompressed images.)", path);
        exit(0);
	}

    // Skip to bytecode
    e = fseek(f, header.bmp_offset, 0);

    // Read the rest of the data pixel by pixel
    unsigned char* pixel = new unsigned char[3];
    float color;

	// Read the pixels for each row and column of Pixels in the image.
	for (int row = 0; row < height; row++)
	{
        for (int col = 0; col < width; col++)
		{
            int index = row * width + col;

            // Read 3 bytes (b, g and r)
            e = fread(pixel, sizeof(unsigned char), 3, f);
            if(e < 0)
            {
                LOG_ERROR("InputFilter::read_bmp: IO error file too short (pixels)");
                exit(0);
            }
        
            // Convert to shade of grey with magic numbers (channel-dependant luminance perception)
            color = pixel[0] * 0.0722f + pixel[1] * 0.7152f + pixel[2] * 0.2126f;
            // Flatten in [0,1]
            color /= 255.0f;

            cache_image->at(index) = color;
        }

        // Rows are padded so that they're always a multiple of 4
		// bytes. This line skips the padding at the end of each row.
        e = fseek(f, width % 4, std::ios::cur);
    }

    fclose(f);
}

void InputFilter::write_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path)
{
    std::ofstream myfile;
    myfile.open("image.txt");
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            myfile << std::format("{:1.1} ", cache_image->at(i * width + j));
        }
        myfile << std::endl;
    }
    myfile.close();
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
            size_t ceil_x = fmin(width - 1, static_cast<size_t>(ceilf(input_x_float)));
            size_t floor_y = static_cast<size_t>(floorf(input_y_float));
            size_t ceil_y = fmin(height - 1, static_cast<size_t>(ceilf(input_y_float)));

            // Get the values of the 4 neighboring pixels
            float v00 = filter_input[floor_y * width + floor_x];
            float v01 = filter_input[floor_y * width + ceil_x];
            float v10 = filter_input[ceil_y * width + floor_x];
            float v11 = filter_input[ceil_y * width + ceil_x];

            // Compute the value of the output pixel using bilinear interpolation
            float q = 0.0f;
            if (ceil_x == floor_x)
            {
                if (ceil_y == floor_y)
                {
                    // Very special case where all the points coincide
                    q = v00;
                }
                else
                {
                    // we can interchange v00 with v01 and v10 with v11 as floor_x and ceil_x coincide
                    q = v00 * (ceil_y - input_y_float) + v10 * (input_y_float - floor_y);
                }
            }
            else
            {
                if (ceil_y == floor_y)
                {
                    // we can interchange v00 with v10 and v01 with v11 as floor_y and ceil_y coincide
                    q = v00 * (ceil_x - input_x_float) + v01 * (input_x_float - floor_x);
                }
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

void InputFilter::interpolate_filter(std::shared_ptr<std::vector<float>> cache_image, size_t fd_width, size_t fd_height)
{
    std::vector<float> copy_filter(cache_image->begin(), cache_image->end());

    cache_image->resize(fd_width * fd_height);
    std::fill(cache_image->begin(), cache_image->end(), 0.0f);

    bilinear_interpolation(copy_filter.data(), cache_image->data(), width, height, fd_width, fd_height);

    width = fd_width;
    height = fd_height;
}

} // namespace holovibes