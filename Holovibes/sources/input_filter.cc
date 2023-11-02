#include "input_filter.hh"

namespace holovibes
{


// Returns the pure image as a char buffer AND sets the width and height of the object
void InputFilter::read_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path)
{
    FILE* f = fopen(path, "rb"); // we do it in pure c because we are S P E E D.
    if(f == NULL)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error");
        exit(0);
    }

    // read the 54-byte header
    unsigned char info[54];
    int e;
    e = fread(info, sizeof(unsigned char), 54, f);
    if(e < 0)
    {
        LOG_ERROR("InputFilter::read_bmp: IO error");
        exit(0);
    }

    // extract image height and width from header
    width = *(int*)&info[18];
    height = *(int*)&info[22];

    // reallocate a byte per pixel
    int size = width * height;
    cache_image->resize(size);

    // read the rest of the data pixel by pixel
    unsigned char* pixel = new unsigned char[3];
    float color;
    for(int i = 0; i < size; i++){
        // Read 3 bytes (r, g and b)
        e = fread(pixel, sizeof(unsigned char), 3, f);
        if(e < 0)
        {
            LOG_ERROR("InputFilter::read_bmp: IO error");
            exit(0);
        }
        // Convert to shade of grey with magic numbers (channel-dependant luminance perception)
        color = pixel[0] * 0.0722f + pixel[1] * 0.7152f + pixel[2] * 0.2126f;
        // Flatten in [0,1]
        color /= 255.0f;

        cache_image->at(i) = color;
    }

    fclose(f);
}

void InputFilter::write_bmp(std::shared_ptr<std::vector<float>> cache_image, const char* path)
{
    for (size_t i = 0; i < height; i++)
    {
        //printf("%.1f ", cache_image->at(i * width));
        /*
        for (size_t j = 0; j < width; j++)
        {
        }
        */
    }
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
            size_t ceil_x = fmin(fd_width - 1, static_cast<size_t>(ceilf(input_x_float)));
            size_t floor_y = static_cast<size_t>(floorf(input_y_float));
            size_t ceil_y = fmin(fd_height - 1, static_cast<size_t>(ceilf(input_y_float)));

            // Get the values of the 4 neighboring pixels
            float v00 = filter_input[floor_y * width + floor_x];
            float v01 = filter_input[floor_y * width + ceil_x];
            float v10 = filter_input[ceil_y * width + floor_x];
            float v11 = filter_input[ceil_y * width + ceil_x];

            // Compute the value of the output pixel using bilinear interpolation
            float q1 = v00 * (ceil_x - input_x_float) + v01 * (input_x_float - floor_x);
            float q2 = v10 * (ceil_x - input_x_float) + v11 * (input_x_float - floor_x);
            float q = q1 * (ceil_y - input_y_float) + q2 * (input_y_float - floor_y);

            // Set the output pixel value
            filter_output[y * fd_width + x] = q;
        }
    }
}

void InputFilter::interpolate_filter(std::shared_ptr<std::vector<float>> cache_image, size_t fd_width, size_t fd_height)
{
    std::vector<float> copy_filter(cache_image->begin(), cache_image->end());

    cache_image->resize(fd_width * fd_height);

    bilinear_interpolation(copy_filter.data(), cache_image->data(), width, height, fd_width, fd_height);

    width = fd_width;
    height = fd_height;
}

} // namespace holovibes