#include "input_filter.hh"

namespace holovibes
{

// Returns the pure image as a char buffer AND sets the width and height of the object
unsigned char* InputFilter::read_bmp(const char* path){
    FILE* f = fopen(path, "rb"); // we do it in pure c because we are S P E E D.
    if (f == NULL){
        LOG_ERROR("Cannot open image");
        exit(0);
    }

    // read the 54-byte header
    unsigned char info[54];
    try
    {
        fread(info, sizeof(unsigned char), 54, f);
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(e.what());
    }

    // extract image height and width from header
    width = *(int*)&info[18];
    height = *(int*)&info[22];

    // allocate 3 bytes per pixel
    int size = 3 * width * height;
    unsigned char* data = new unsigned char[size];

    // read the rest of the data at once
    try
    {
        fread(data, sizeof(unsigned char), size, f); 
    }
    catch(const std::exception& e)
    {
        LOG_ERROR(e.what());
    }

    fclose(f);

    return data;
}

void InputFilter::normalize_filter(const cudaStream_t stream) {}

void InputFilter::interpolate_filter(size_t fd_width, size_t fd_height, const cudaStream_t stream) {}

void InputFilter::apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream) {}
}