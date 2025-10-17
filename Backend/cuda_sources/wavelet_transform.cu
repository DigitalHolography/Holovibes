#include "wavelet_transform.cuh"
#include "complex_utils.cuh"
#include "morlet_wavelet.cuh"
#include "frame_desc.hh"

using camera::FrameDescriptor;


__global__ void mul_conj_kernel(const cuComplex* X, const cuComplex* Psi, cuComplex* Y, int N)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= N) return;

    float xr = X[k].x, xi = X[k].y;
    float pr = Psi[k].x, pi = Psi[k].y;

    // multiply by conjugate of wavelet
    Y[k].x = xr * pr + xi * pi;
    Y[k].y = xi * pr - xr * pi;
}

__global__ void scale_ifft_kernel(cuComplex* data, int N)
{
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= N) return;
    float s = 1.0f / float(N);
    data[k].x *= s;
    data[k].y *= s;
}

void wavelet_transform(cuComplex* output, cuComplex* input, const cufftHandle plan1d,  const FrameDescriptor& fd, int tranformation_size)
{
    int N =  tranformation_size; // number of samples
    float fs = 1; // sampling rate in Hz (fps) dummy for now
    float dt = 1.0f / fs;
    float scale = 0.05f;      // pick a scale
    float omega0 = 6.0f;      // central frequency (standard)

    // allocate buffer on GPU
    cuComplex* d_kernel;
    cudaMalloc(&d_kernel, N * sizeof(cuComplex));

    // fill with Morlet kernel
    createMorletKernel(d_kernel, N, dt, scale, omega0);
    // now d_kernel holds the wavelet in frequency domain, ready to multiply with FFT(signal)

    // FFT input in-place
    cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);

    // Multiply by wavelet in frequency domain
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mul_conj_kernel<<<blocks, threads>>>(input, d_kernel, output, N);

    // IFFT back to time domain (in-place in output)
    cufftExecC2C(plan1d, output, output, CUFFT_INVERSE);

    // Scale by 1/N to normalize FFT/IFFT
    scale_ifft_kernel<<<blocks, threads>>>(output, N);

    cudaFree(d_kernel);
}

