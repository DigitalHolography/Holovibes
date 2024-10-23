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

// Kernel pour appliquer un flou gaussien 5x5
__global__ void gaussian_blur_5x5(const float* input, float* output, int width, int height, const float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Sortir si en dehors des limites

    float sum = 0.0f;
    int halfKernel = 2;  // Noyau 5x5 a une moitié de taille 2 (de -2 à 2 autour du centre)

    // Appliquer le noyau gaussien
    for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
        for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
            int ix = min(max(x + kx, 0), width - 1);  // Clamper les indices dans l'image
            int iy = min(max(y + ky, 0), height - 1);

            // Calcul de l'indice dans l'image et application du noyau
            float pixelValue = input[iy * width + ix];
            float kernelValue = kernel[(ky + halfKernel) * 5 + (kx + halfKernel)];
            sum += pixelValue * kernelValue;
        }
    }

    // Écriture du résultat dans l'image de sortie
    output[y * width + x] = sum;
}

// Fonction pour créer un noyau gaussien 5x5
void create_gaussian_kernel_5x5(float* kernel, float sigma) {
    const int size = 5;
    int halfSize = size / 2;
    float sum = 0.0f;
    float sigma2 = 2.0f * sigma * sigma;

    // Générer les valeurs du noyau gaussien
    for (int y = -halfSize; y <= halfSize; ++y) {
        for (int x = -halfSize; ++x <= halfSize;) {
            float value = expf(-(x * x + y * y) / sigma2) / (M_PI * sigma2);
            kernel[(y + halfSize) * size + (x + halfSize)] = value;
            sum += value;
        }
    }

    // Normaliser le noyau pour que la somme soit égale à 1
    for (int i = 0; i < size * size; ++i) {
        kernel[i] /= sum;
    }
}

// Fonction pour appliquer le flou gaussien 5x5 avec CUDA
void fast_gaussian_blur(float* d_input, float* d_output, int width, int height, float gw, cudaStream_t stream = 0) {
    // Création du noyau gaussien sur l'hôte
    float h_kernel[25];  // 5x5 = 25 éléments
    create_gaussian_kernel_5x5(h_kernel, gw);

    // Allocation de mémoire pour le noyau gaussien sur le GPU
    float* d_kernel;
    cudaMalloc(&d_kernel, 25 * sizeof(float));
    cudaMemcpyAsync(d_kernel, h_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Définir la taille du bloc et de la grille
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancer le kernel CUDA pour appliquer le flou gaussien
    gaussian_blur_5x5<<<numBlocks, threadsPerBlock, 0, stream>>>(d_input, d_output, width, height, d_kernel);

    // Synchroniser pour attendre la fin du kernel
    cudaStreamSynchronize(stream);

    // Libérer la mémoire du noyau gaussien sur le GPU
    cudaFree(d_kernel);
}



__global__ void kernel_normalize(float* in_out, const uint min, const uint max, const uint size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        in_out[index] = (in_out[index] - min) / (max - min);
}

__global__ void kernel_multiplication(float* in_out, const float value, const uint size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        in_out[index] *= value;
}

__global__ void kernel_division(float* in_out, const float* value, const uint size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
        in_out[index] = value[index];
}


// Kernel de réduction pour calculer la somme
__global__ void sum_region_kernel(const float* input, float* output, int width, int a, int b, int c, int d) {
    extern __shared__ float sdata[]; // mémoire partagée pour la réduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialiser la mémoire partagée avec des zéros
    sdata[tid] = 0;

    // Vérifier que l'index global est dans les limites de la région spécifiée
    int x = idx % width;
    int y = idx / width;

    if (x >= a && x <= b && y >= c && y <= d) {
        sdata[tid] = input[idx];
    }

    // Synchroniser tous les threads du bloc avant la réduction
    __syncthreads();

    // Réduction en utilisant une addition parallèle
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Le thread 0 stocke le résultat partiel pour ce bloc
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

void apply_flat_field_correction(float* input_output, const uint width, const float gw, const float borderAmount, const cudaStream_t stream) {
    int size = width * width;
    // Trouver le min et max de l'image
    float* h_image = new float[size];
    cudaXMemcpyAsync(h_image, input_output, size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);
    auto Im_min = *std::min_element(h_image, h_image + size);
    //std::cout << "le min : " << Im_min << std::endl;
    auto Im_max = *std::max_element(h_image, h_image + size);
    //std::cout << "le max : " << Im_max << std::endl;


    bool flag = false;
    if (Im_min < 0 || Im_max > 1) {
        flag = true;
        uint threads = get_max_threads_1d();
        uint blocks = map_blocks_to_problem(size, threads);
        kernel_normalize<<<blocks, threads, 0, stream>>>(input_output, Im_min, Im_max, size);
        cudaXStreamSynchronize(stream);
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
    
    // Allocation de mémoire pour stocker le résultat partiel de la somme
    float* d_sum;
    float h_sum = 0;
    cudaXMalloc(&d_sum, sizeof(float));
    cudaXMemcpyAsync(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice, stream);

    // Lancer le kernel pour calculer la somme dans la zone définie par (a, b, c, d)
    uint threads = get_max_threads_1d();
    uint blocks = map_blocks_to_problem(size, threads);
    size_t sharedMemSize = threads * sizeof(float);  // Taille de la mémoire partagée
    sum_region_kernel<<<blocks, threads, sharedMemSize, stream>>>(input_output, d_sum, width, a, b, c, d);
    cudaXStreamSynchronize(stream);  // Attendre que la somme soit calculée

    // Copier la somme du GPU vers l'hôte
    cudaXMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);

    float ms = h_sum;
    
    float* d_input_blur;
    cudaXMalloc(&d_input_blur, size * sizeof(float));
    fast_gaussian_blur(input_output, d_input_blur, width, width, gw, stream);

    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(size, threads);
    kernel_division<<<blocks, threads, 0, stream>>>(input_output, d_input_blur, size);
    cudaXStreamSynchronize(stream);
    
    /*// Allocation de mémoire pour stocker le résultat partiel de la somme
    float* d_sum2;
    float h_sum2 = 0;
    cudaXMalloc(&d_sum2, sizeof(float));
    cudaXMemcpyAsync(d_sum2, &h_sum2, sizeof(float), cudaMemcpyHostToDevice, stream);

    // Lancer le kernel pour calculer la somme dans la zone définie par (a, b, c, d)
    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(size, threads);
    sharedMemSize = threads * sizeof(float);  // Taille de la mémoire partagée
    sum_region_kernel<<<blocks, threads, sharedMemSize, stream>>>(input_output, d_sum2, width, a, b, c, d);
    cudaXStreamSynchronize(stream);  // Attendre que la somme soit calculée

    // Copier la somme du GPU vers l'hôte
    cudaXMemcpyAsync(&h_sum2, d_sum2, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaXStreamSynchronize(stream);

    float ms2 = h_sum2;

    std::cout << "ms / ms2 : " << ms / ms2 << std::endl;
    std::cout << "ms : " << ms << std::endl;
    threads = get_max_threads_1d();
    blocks = map_blocks_to_problem(size, threads);
    kernel_multiplication<<<blocks, threads, 0, stream>>>(input_output, ms / ms2, size);
    cudaXStreamSynchronize(stream);
    
    /*
    if (flag)
    {
        for (int i = 0; i < size; i++)
        {
            copy_input[i] = Im_min + (Im_max - Im_min) * copy_input[i];
        }
    }*/
    cudaXStreamSynchronize(stream);

    delete[] h_image;
    cudaXFree(d_sum);
    //cudaXFree(d_sum2);
    cudaXFree(d_input_blur);
}
