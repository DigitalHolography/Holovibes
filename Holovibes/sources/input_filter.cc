#include "input_filter.hh"

namespace holovibes
{

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

void InputFilter::normalize_filter(const cudaStream_t stream) {}

void InputFilter::interpolate_filter(size_t fd_width, size_t fd_height, const cudaStream_t stream) {}

void InputFilter::apply_filter(cuComplex* gpu_input, size_t fd_width, size_t fd_height, const cudaStream_t stream) {}
}