#include "hardware_limits.hh"
#include "wavelet_transform.cuh"
#include "complex_utils.cuh"
#include "morlet_wavelet.cuh"
#include "frame_desc.hh"
#include <iostream>
using camera::FrameDescriptor;


__global__ void mul_conj_kernel(const cuComplex* X,
                                const cuComplex* Psi,
                                cuComplex* Y,
                                int N,
                                int total)    
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int frame_res = total / N;
    int k = idx / frame_res; // freq bin within the transform
    float xr = X[idx].x, xi = X[idx].y;
    float pr = Psi[k].x, pi = Psi[k].y;

    // multiply X[idx] by conjugate(Psi[k])
    Y[idx].x = xr * pr + xi * pi;
    Y[idx].y = xi * pr - xr * pi; 
}

__global__ void scale_kernel(cuComplex* data, int total_elements)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= total_elements) return;
    float s = 1.0f / float(total_elements);
    data[k].x *= s;
    data[k].y *= s;
}

void wavelet_transform(cuComplex* output, cuComplex* input, const cufftHandle plan1d,  const FrameDescriptor& fd, int tranformation_size, const cudaStream_t stream, float target_freq)
{
    int N =  tranformation_size;
    float dt = 1.0f; // time step can be set to 1.0 as we work in normalized units (we dont care about the actual time scale here)
    float omega0 = 6.0f; // central frequency (standard)
    float scale = omega0 / (2.0f * M_PI * target_freq);

    cuComplex* d_kernel;
    cudaMalloc(&d_kernel, N * sizeof(cuComplex));

    createMorletKernel(d_kernel, N, dt, scale, omega0, stream);

    cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);

    // Multiply by wavelet in frequency domain
    int threads = get_max_threads_1d();
    int total = N * fd.get_frame_res();

    int blocks = map_blocks_to_problem(total, threads);

    mul_conj_kernel<<<blocks, threads, 0, stream>>>(input, d_kernel, output, N, total);

    // Scale by 1/N to normalize FFT
    scale_kernel<<<blocks, threads, 0,stream>>>(output, N);

    cudaFree(d_kernel);
}

