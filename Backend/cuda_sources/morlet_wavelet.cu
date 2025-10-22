#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "complex_utils.cuh"
#include "morlet_wavelet.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// Kernel to compute Morlet wavelet in frequency domain  (simplified version i dont substract the correction terms)
__global__ void buildMorletKernel(cuComplex* kernel,
                                  int N, float dt,
                                  float scale, float omega0)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;
    
    float freq_hz;
    if (k < N/2 + 1) {
        // Positive frequencies (0 to N/2)
        freq_hz = k / (N * dt);
    } else {
        // Negative frequencies (-N/2+1 to -1)
        freq_hz = (k - N) / (N * dt);
    }
   
    float omega = 2.0f * M_PI * freq_hz;
    float arg = scale * omega - omega0;
    float value = sqrtf(scale) * expf(-0.5f * arg * arg);
    
    // FIX: Use symmetric wavelet for real-valued output
    // This preserves both positive and negative frequencies
    kernel[k] = make_cuComplex(value, 0.0f);
}

void createMorletKernel(cuComplex* d_kernel, int N, float dt,
                        float scale, float omega0, const cudaStream_t stream)
{
    uint threads = get_max_threads_1d();
    uint blocks  = map_blocks_to_problem(N, threads);

    buildMorletKernel<<<blocks, threads, 0, stream>>>(d_kernel, N, dt, scale, omega0);
}
