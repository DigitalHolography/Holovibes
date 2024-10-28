#include "cuda_memory.cuh"
#include "common.cuh"
#include "tools_analysis.cuh"
#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"

void load_kernel_in_GPU(cuComplex* output, const float* kernel, const size_t frame_res, cudaStream_t stream)
{
   cudaMemcpy2DAsync(output,
                                sizeof(cuComplex),
                                kernel,
                                sizeof(float),
                                sizeof(float),
                                frame_res,
                                cudaMemcpyHostToDevice,
                                stream);
}

float* kernel_add_padding(float* kernel, const int width, const int height, const int new_width, const int new_height) {
    // Check that new dimensions are greater than or equal to the original dimensions
    if (new_width < width || new_height < height) {
        std::cerr << "New dimensions must be greater than or equal to the original dimensions." << std::endl;
        return nullptr;
    }

    // Create a new array for the padded kernel, initialized to 0
    float* padded_kernel = new float[new_width * new_height];
    std::memset(padded_kernel, 0, new_width * new_height * sizeof(float));

    // Calculate the starting position (top-left corner) of the original kernel in the padded kernel
    int start_x = (new_width - width) / 2;
    int start_y = (new_height - height) / 2;

    // Copy the original kernel into the center of the new padded array
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Map original kernel indices to the padded array indices
            padded_kernel[(start_y + y) * new_width + (start_x + x)] = kernel[y * width + x];
        }
    }

    return padded_kernel;
}