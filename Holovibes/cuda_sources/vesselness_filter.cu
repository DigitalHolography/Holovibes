#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"

float comp_hermite_rec(int n, float x)
{
    if (n == 0)
        return 1.0f;
    else if (n == 1)
        return 2.0f * x;
    else if (n > 1)
        return (2.0f * x * comp_hermite_rec(n - 1, x)) - (2 * (n - 1) * comp_hermite_rec(n - 2, x));
    else
        throw std::exception("comp_hermite_rec in velness_filter.cu : n can't be negative");
}

float* comp_gaussian(float* x, float sigma, int x_size)
{
    float *to_ret = new float [x_size];
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
    return C;
}

float* gaussian_imfilter_sep(float* input_img, 
                            float* input_x, 
                            float* input_y, 
                            const size_t frame_res, 
                            float* convolution_buffer, 
                            cuComplex* cuComplex_buffer, 
                            CufftHandle* convolution_plan, 
                            cudaStream_t stream)
{
    // Copy image
    float* gpu_output;
    cudaXMalloc(&gpu_output, frame_res * sizeof(float));
    cudaXMemcpy(gpu_output, input_img, frame_res * sizeof(float));

    cuComplex* output_complex_x;
    cudaXMalloc(&output_complex_x, frame_res * sizeof(cuComplex));
    load_kernel_in_GPU(output_complex_x, input_x, frame_res, stream);

    cuComplex* output_complex_y;
    cudaXMalloc(&output_complex_y, frame_res * sizeof(cuComplex));
    load_kernel_in_GPU(output_complex_y, input_y, frame_res, stream);
    // Apply both convolutions
    convolution_kernel(gpu_output,
                        convolution_buffer,
                        cuComplex_buffer,
                        convolution_plan,
                        frame_res,
                        output_complex_x,
                        true,
                        stream);
    convolution_kernel(gpu_output,
                        convolution_buffer,
                        cuComplex_buffer,
                        convolution_plan,
                        frame_res,
                        output_complex_y,
                        true,
                        stream);


    float *res = new float[frame_res];
    cudaXMemcpy(res, gpu_output, frame_res * sizeof(float), cudaMemcpyDeviceToHost);

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
                        float* g_xx_px, 
                        float* g_xx_qy, 
                        float* g_xy_px, 
                        float* g_xy_qy, 
                        float* g_yy_px, 
                        float* g_yy_qy, 
                        int frame_size, 
                        float* convolution_buffer, 
                        cuComplex* cuComplex_buffer,
                        CufftHandle* convolution_plan,
                        cudaStream_t stream)
{
    int gamma = 1;

    float A = std::pow(sigma, gamma);

    float* Ixx = gaussian_imfilter_sep(input, g_xx_px, g_xx_qy, frame_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Ixx, A, frame_size);

    float* Ixy = gaussian_imfilter_sep(input, g_xx_px, g_xx_qy, frame_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Ixy, A, frame_size);

    float* Iyx = new float[frame_size];
    for (size_t i = 0; i < frame_size; ++i) {
        Iyx[i] = Ixy[i];
    }

    float* Iyy = gaussian_imfilter_sep(input, g_xx_px, g_xx_qy, frame_size, convolution_buffer, cuComplex_buffer, convolution_plan, stream);
    multiply_by_float(Iyy, A, frame_size);

    return Iyy;
}