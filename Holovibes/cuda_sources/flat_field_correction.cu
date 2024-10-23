#include "filter2D.cuh"
#include "shift_corners.cuh"
#include "apply_mask.cuh"
#include "tools_conversion.cuh"
#include "cuda_memory.cuh"
/*
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
*/
#include <cuda_runtime.h>
#include <iostream>

#include <cufftXt.h>

__global__ void normalizeImage(float* image, float Im_min, float Im_max, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        image[index] = (image[index] - Im_min) / (Im_max - Im_min);
    }
}

__global__ void denormalizeImage(float* image, float Im_min, float Im_max, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        image[index] = image[index] * (Im_max - Im_min) + Im_min;
    }
}

__global__ void normalizeByGaussian(float* image, float* blurredImage, float* output, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = image[index] / (blurredImage[index] + 1e-6);
    }
}

__global__ void applyGaussianBlur(float* input, float* output, int width, int height, const float* kernel, int kWidth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_k = kWidth / 2;

    if (x < width && y < height) {
        float sum = 0.0f;
        float weightSum = 0.0f;

        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                float weight = kernel[(ky + half_k) * kWidth + (kx + half_k)];
                sum += input[iy * width + ix] * weight;
                weightSum += weight;
            }
        }
        output[y * width + x] = sum / weightSum;
    }
}

//!
//! \fn void std_to_box(int boxes[], float sigma, int n)  
//!
//! \brief this function converts the standard deviation of 
//! Gaussian blur into dimensions of boxes for box blur. For 
//! further details please refer to :
//! https://www.peterkovesi.com/matlabfns/#integral
//! https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
//!
//! \param[out] boxes   boxes dimensions
//! \param[in] sigma    Gaussian standard deviation
//! \param[in] n        number of boxes
//!
void std_to_box(int boxes[], float sigma, int n)  
{
    // ideal filter width
    float wi = std::sqrt((12*sigma*sigma/n)+1); 
    int wl = std::floor(wi);  
    if(wl%2==0) wl--;
    int wu = wl+2;
                
    float mi = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4);
    int m = std::round(mi);
                
    for(int i=0; i<n; i++) 
        boxes[i] = ((i < m ? wl : wu) - 1) / 2;
}

//!
//! \fn void horizontal_blur(float * in, float * out, int w, int h, int r)    
//!
//! \brief this function performs the horizontal blur pass for box blur. 
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] r            box dimension
//!
void horizontal_blur(float * in, float * out, int w, int h, int r) 
{
    float iarr = 1.f / (r+r+1);
    #pragma omp parallel for
    for(int i=0; i<h; i++) 
    {
        int ti = i*w, li = ti, ri = ti+r;
        float fv = in[ti], lv = in[ti+w-1], val = (r+1)*fv;

        for(int j=0; j<r; j++) val += in[ti+j];
        for(int j=0  ; j<=r ; j++) { val += in[ri++] - fv      ; out[ti++] = val*iarr; }
        for(int j=r+1; j<w-r; j++) { val += in[ri++] - in[li++]; out[ti++] = val*iarr; }
        for(int j=w-r; j<w  ; j++) { val += lv       - in[li++]; out[ti++] = val*iarr; }
    }
}

//!
//! \fn void total_blur(float * in, float * out, int w, int h, int r)   
//!
//! \brief this function performs the total blur pass for box blur. 
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] r            box dimension
//!
void total_blur(float * in, float * out, int w, int h, int r) 
{
    float iarr = 1.f / (r+r+1);
    #pragma omp parallel for
    for(int i=0; i<w; i++) 
    {
        int ti = i, li = ti, ri = ti+r*w;
        float fv = in[ti], lv = in[ti+w*(h-1)], val = (r+1)*fv;
        for(int j=0; j<r; j++) val += in[ti+j*w];
        for(int j=0  ; j<=r ; j++) { val += in[ri] - fv    ; out[ti] = val*iarr; ri+=w; ti+=w; }
        for(int j=r+1; j<h-r; j++) { val += in[ri] - in[li]; out[ti] = val*iarr; li+=w; ri+=w; ti+=w; }
        for(int j=h-r; j<h  ; j++) { val += lv     - in[li]; out[ti] = val*iarr; li+=w; ti+=w; }
    }
}

//!
//! \fn void box_blur(float * in, float * out, int w, int h, int r)    
//!
//! \brief this function performs a box blur pass. 
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] r            box dimension
//!
void box_blur(float *& in, float *& out, int w, int h, int r) 
{
    std::swap(in, out);
    horizontal_blur(out, in, w, h, r);
    total_blur(in, out, w, h, r);
    // Note to myself : 
    // here we could go anisotropic with different radiis rx,ry in HBlur and TBlur
}

//!
//! \fn void fast_gaussian_blur(float * in, float * out, int w, int h, float sigma)   
//!
//! \brief this function performs a fast Gaussian blur. Applying several
//! times box blur tends towards a true Gaussian blur. Three passes are sufficient
//! for good results. For further details please refer to :  
//! http://blog.ivank.net/fastest-gaussian-blur.html
//!
//! \param[in,out] in       source channel
//! \param[in,out] out      target channel
//! \param[in] w            image width
//! \param[in] h            image height
//! \param[in] r            box dimension
//!
void fast_gaussian_blur(float *& in, float *& out, int w, int h, float sigma) 
{
    // sigma conversion to box dimensions
    int boxes[3];
    std_to_box(boxes, sigma, 3);
    box_blur(in, out, w, h, boxes[0]);
    box_blur(out, in, w, h, boxes[1]);
    box_blur(in, out, w, h, boxes[2]);
}

void apply_flat_field_correction(float* input_output, const uint width, const float gw, const float borderAmount, const cudaStream_t stream) {
    int size = width * width;
    // Trouver le min et max de l'image
    float* h_image = new float[size];
    float* copy_input = new float[size];
    cudaXMemcpyAsync(h_image, input_output, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXMemcpyAsync(copy_input, input_output, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    auto Im_min = *std::min_element(h_image, h_image + size);
    auto Im_max = *std::max_element(h_image, h_image + size);

    bool flag = false;
    if (Im_min < 0 || Im_max > 1) {
        flag = true;
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        for (int i = 0; i < size; i++)
        {
            copy_input[i] = (copy_input[i] - Im_min) / (Im_max - Im_min);
        }
    }
    
    int a=0, b=0, c=0, d=0;
    if (borderAmount == 0)
    {
        a = 1;
        b = width;
        c = 1;
        d = width;
    }
    else 
    {
        a = std::ceil(width * borderAmount);
        b = std::floor(width * ( 1 - borderAmount));
        c = std::ceil(width * borderAmount);
        d = std::floor(width * ( 1 -  borderAmount));
    }
    
    int a_bis = a;
    int c_bis = c;
    float sum = 0;
    while (a_bis <= b)
    {
        while (c_bis <= d)
        {
            sum += copy_input[a_bis + c_bis * width];
            c_bis++;
        }
        a_bis++;
    }
    float ms = sum;

    cudaXMemcpyAsync(h_image, input_output, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    fast_gaussian_blur(h_image, copy_input, width, width, gw);
    
    a_bis = a;
    c_bis = c;
    sum = 0;
    while (a_bis <= b)
    {
        while (c_bis <= d)
        {
            sum += copy_input[a_bis + c_bis * width];
            c_bis++;
        }
        a_bis++;
    }
    float ms2 = sum;

    for (int i = 0; i < size; i++)
    {
        copy_input[i] *= (ms / ms2);
    }
    
    if (flag)
    {
        for (int i = 0; i < size; i++)
        {
            copy_input[i] = Im_min + (Im_max - Im_min) * copy_input[i];
        }
    }
    cudaXStreamSynchronize(stream);

    cudaXMemcpyAsync(input_output, copy_input, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    delete[] h_image;
}
