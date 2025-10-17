#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "complex_utils.cuh"
#include "morlet_wavelet.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



// Kernel to compute Morlet wavelet in frequency domain
__global__ void buildMorletKernel(cuComplex* kernel, 
                                  int N, float dt, 
                                  float scale, float omega0) 
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    // Compute frequency bin (Hz)
    float freq_hz;
    if (k <= N/2) {
        freq_hz = k / (N * dt);
    } else {
        freq_hz = -(N - k) / (N * dt);
    }
    float omega = 2.0f * M_PI * freq_hz; // angular frequency

    // Morlet spectrum: Gaussian centered at omega0
    float arg = scale * omega - omega0;
    float value = expf(-0.5f * arg * arg);

    // Make it analytic: zero negative frequencies
    if (freq_hz < 0.0f) {
        value = 0.0f;
    }

    // Scaling by sqrt(scale) (continuous wavelet convention)
    value *= sqrtf(scale);

    // Result is purely real (imag = 0)
    kernel[k] = make_cuComplex(value, 0.0f);
}


void createMorletKernel(cuComplex* d_kernel, int N, float dt,
                        float scale, float omega0)
{
    uint threads = get_max_threads_1d();
    uint blocks  = map_blocks_to_problem(N, threads);

    buildMorletKernel<<<blocks, threads>>>(d_kernel, N, dt, scale, omega0);
    cudaDeviceSynchronize();
}
