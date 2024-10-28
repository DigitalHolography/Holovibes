#include <iostream>
#include <fstream>

#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"

static void write1DFloatArrayToFile(const float* array, int rows, int cols, const std::string& filename)
{
    // Open the file in write mode
    std::ofstream outFile(filename);

    // Check if the file was opened successfully
    if (!outFile)
    {
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return;
    }

    // Write the 1D array in row-major order to the file
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            outFile << array[i * cols + j]; // Calculate index in row-major order
            if (j < cols - 1)
            {
                outFile << " "; // Separate values in a row by a space
            }
        }
        outFile << std::endl; // New line after each row
    }

    // Close the file
    outFile.close();
    std::cout << "1D array written to the file " << filename << std::endl;
}

float comp_hermite_rec(int n, float x)
{
    if (n == 0)
        return 1.0f;
    if (n == 1)
        return 2.0f * x;
    if (n > 1)
        return (2.0f * x * comp_hermite_rec(n - 1, x)) - (2.0f * (n - 1) * comp_hermite_rec(n - 2, x));
    throw std::exception("comp_hermite_rec in velness_filter.cu : n can't be negative");
}

float* comp_gaussian(float* x, float sigma, int x_size)
{
    float *to_ret = new float[x_size];
    for (int i = 0; i < x_size; i++)
    {
        to_ret[i] = (1 / (sigma * (sqrt(2 * M_PI))) * std::exp((-1 * (std::pow(x[i], 2))) / (2 * std::pow(sigma, 2))));
    }
    return to_ret;
}

float* comp_dgaussian(float* x, float sigma, int n, int x_size)
{
    float A = std::pow((-1 / (sigma * std::sqrt(2))), n);
    float *B = new float[x_size];
    for (int i = 0; i < x_size; i++)
    {
        B[i] = comp_hermite_rec(n, x[i] / (sigma * std::sqrt(2)));
    }
    float* C = comp_gaussian(x, sigma, x_size);
    for (int i = 0; i < x_size; i++)
    {
        C[i] = A * B[i] * C[i];   
    }
    delete[] B;
    return C;
}

float* gaussian_imfilter_sep(float* input_img, 
                            float* kernel,
                            const size_t frame_res, 
                            float* convolution_buffer, 
                            cuComplex* cuComplex_buffer, 
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
   
    // Copy image
    float* gpu_output;
    cudaXMalloc(&gpu_output, frame_res * sizeof(float));
    cudaXMemcpyAsync(gpu_output, input_img, frame_res * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    cudaXStreamSynchronize(stream);

    cuComplex* output_complex;
    cudaXMalloc(&output_complex, frame_res * sizeof(cuComplex));
    load_kernel_in_GPU(output_complex, kernel, frame_res, stream);
    cudaXStreamSynchronize(stream);

    // Apply both convolutions
    convolution_kernel(gpu_output,
                        convolution_buffer,
                        cuComplex_buffer,
                        convolution_plan,
                        frame_res,
                        output_complex,
                        false,
                        stream);
    cudaXStreamSynchronize(stream);


    // convolution_kernel(gpu_output,
    //                     convolution_buffer,
    //                     cuComplex_buffer,
    //                     convolution_plan,
    //                     frame_res,
    //                     output_complex_y,
    //                     true,
    //                     stream);

    // cudaXStreamSynchronize(stream);
    float *res = new float[frame_res];
    cudaXMemcpyAsync(res, gpu_output, frame_res * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    write1DFloatArrayToFile(res, std::sqrt(frame_res), std::sqrt(frame_res), "convo.txt");

    return res;
}

void multiply_by_float(float* vect, float num, int frame_size)
{
    for (int i = 0; i < frame_size; i++)
    {
        vect[i] *= num;
    }
}


float* vesselness_filter(float* input, 
                        float sigma, 
                        float* g_xx_mul, 
                        float* g_xy_mul, 
                        float* g_yy_mul, 
                        int frame_res, 
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cudaStream_t stream)
{
    int gamma = 1;

    float A = std::pow(sigma, gamma);

    float* Ixx = gaussian_imfilter_sep(input, g_xx_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Ixx, A, frame_res);

    float* Ixy = gaussian_imfilter_sep(input, g_xy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Ixy, A, frame_res);

    float* Iyx = new float[frame_res];
    for (size_t i = 0; i < frame_res; ++i) {
        Iyx[i] = Ixy[i];
    }

    float* Iyy = gaussian_imfilter_sep(input, g_yy_mul, frame_res, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Iyy, A, frame_res);

    return Iyy;
}