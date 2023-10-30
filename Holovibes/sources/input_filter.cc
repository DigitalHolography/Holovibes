#include "input_filter.hh"

namespace holovibes
{


// Returns the pure image as a char buffer AND sets the width and height of the object
unsigned char* InputFilter::read_bmp(const char* path)
{
    FILE* f = fopen(path, "rb"); // we do it in pure c because we are S P E E D.
    if (f == NULL)
    {
        LOG_ERROR("Cannot open image");
        exit(0);
    }

    // read the 54-byte header
    unsigned char info[54];
    try
    {
        fread(info, sizeof(unsigned char), 54, f);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR(e.what());
    }

    // extract image height and width from header
    width = *(int*)&info[18];
    height = *(int*)&info[22];

    // allocate a byte per pixel
    int size = width * height;
    unsigned char* data = new unsigned char[size];

    // read the rest of the data at once

    unsigned char* pixel = new unsigned char[3];
    for(int i = 0; i < size; i++){
        try
        {
            fread(pixel, sizeof(unsigned char), 3, f); 
            data[i] = ((pixel[0] + pixel[1] + pixel[2]) / 3);
        }
        catch(const std::exception& e)
        {
            LOG_ERROR(e.what());
        }
    }

    fclose(f);

    return data;
}

void get_min_max(float* filter, size_t frame_res, float* min, float* max)
{
    for (size_t i = 0; i < frame_res; i++)
    {
        float val = filter[i];
        if (val > *max)
            *max = val;
        if (val < *min)
            *min = val;
    }
}

void InputFilter::normalize_filter(float* filter, size_t frame_res)
{
    float min = 1.0f;
    float max = 0.0f;
    get_min_max(filter, frame_res, &min, &max);

    for (size_t i = 0; i < frame_res; i++)
    {
        filter[i] = (filter[i] - min) / (max - min);
    }
}

void InputFilter::interpolate_filter(
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
            size_t ceil_x = min(fd_width - 1, static_cast<size_t>(ceilf(input_x_float)));
            size_t floor_y = static_cast<size_t>(floorf(input_y_float));
            size_t ceil_y = min(fd_height - 1, static_cast<size_t>(ceilf(input_y_float)));

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

void InputFilter::apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream) {}

} // namespace holovibes