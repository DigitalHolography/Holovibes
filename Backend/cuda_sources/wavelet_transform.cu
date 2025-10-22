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

__global__ void scale_ifft_kernel(cuComplex* data, int total_elements)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= total_elements) return;
    float s = 1.0f / float(total_elements); // or use transformation_size
    data[k].x *= s;
    data[k].y *= s;
}

__global__ void fill_gradient(cuComplex* data, int total, float start_val = 0.0f, float end_val = 1.0f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Compute gradient factor (0.0 -> 1.0)
    float t = float(idx) / float(total - 1);

    // Interpolate linearly between start_val and end_val
    float val = start_val + t * (end_val - start_val);

    // Fill both real and imaginary parts with the same value (or just real if you want)
    data[idx] = make_cuComplex(val, val);
}

void wavelet_transform(cuComplex* output, cuComplex* input, const cufftHandle plan1d,  const FrameDescriptor& fd, int tranformation_size, const cudaStream_t stream)
{
    LOG_ERROR("Dam wavelet");
    int N =  tranformation_size; // number of frames in time dimension
    float fs = 1; // sampling rate in Hz (fps) dummy for now
    float dt = 1.0f / fs;
    float omega0 = 6.0f;      // central frequency (standard)
    float target_freq = 2.0f; 
    float scale = omega0 / (2.0f * M_PI * target_freq);
    printf("Wavelet parameters: N=%d, fs=%.1f, dt=%.4f, scale=%.4f, target_freq=%.2f Hz\n", N, fs, dt, scale, target_freq);

    // allocate buffer on GPU
    cuComplex* d_kernel;
    cudaMalloc(&d_kernel, N * sizeof(cuComplex));

    // fill with Morlet kernel
    createMorletKernel(d_kernel, N, dt, scale, omega0, stream);
    // now d_kernel holds the wavelet in frequency domain, ready to multiply with FFT(signal)

    // FFT input in-place
    cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);

    // Multiply by wavelet in frequency domain
    int threads = get_max_threads_1d();
    int total = N * fd.get_frame_res();

    int blocks = map_blocks_to_problem(total, threads);
    printf("Wavelet parameters: total=%d, threads=%d, blocks=%d\n", total, threads, blocks);

    LOG_ERROR("Dam wavelet before kernel");
    mul_conj_kernel<<<blocks, threads, 0, stream>>>(input, d_kernel, output, N, total);
    LOG_ERROR("Dam wavelet after kernel");

    // Scale by 1/N to normalize FFT/IFFT
    scale_ifft_kernel<<<blocks, threads, 0,stream>>>(output, N);

    LOG_ERROR("Dam wavelet asfter fill ");

  
    cudaFree(d_kernel);
}

