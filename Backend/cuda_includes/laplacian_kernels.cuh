#ifndef LAPLACIAN_KERNELS_CUH
#define LAPLACIAN_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdint>

// Kernel CUDA pour calculer le Laplacien sur une image 8 bits.
// Pour chaque pixel (hors bords), on calcule :
//   lap = I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4*I(x,y)
__global__ void laplacianKernel8(const uint8_t* input, float* output, int width, int height);

// Kernel CUDA pour calculer le Laplacien sur une image 16 bits.
__global__ void laplacianKernel16(const uint16_t* input, float* output, int width, int height);

// Functor pour calculer le carré d'une valeur, utilisé dans la réduction.
struct square
{
    __host__ __device__ float operator()(const float& x) const;
};

float processFrameCUDA(const void* frameData, int width, int height, int depth);

#endif // LAPLACIAN_KERNELS_CUH