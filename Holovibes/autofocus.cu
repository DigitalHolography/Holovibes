/*! \file */
#include "autofocus.cuh"

# include <cuda_runtime.h>
# include "device_launch_parameters.h"
# include "hardware_limits.hh"
# include "tools.cuh"
# include "tools_multiply.cuh"
# include "tools_divide.cuh"
# include "average.cuh"
# include <stdio.h>

static __global__ void kernel_minus_operator(
  const float* input_left,
  const float* input_right,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input_left[index] - input_right[index];
    index += blockDim.x * gridDim.x;
  }
}

static float global_variance_intensity(
  const float* input,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  // <I>
  const float average_input = average_operator(input, size);

  // We create a matrix of <I> in order to do the substraction
  float* matrix_average;
  cudaMalloc(&matrix_average, size * sizeof(float));

  float* cpu_average_matrix = (float *)malloc(sizeof(float) * size);
  for (unsigned int i = 0; i < size; ++i)
    cpu_average_matrix[i] = average_input;

  cudaMemcpy(matrix_average, cpu_average_matrix, size * sizeof(float), cudaMemcpyHostToDevice);

  // I - <I>
  kernel_minus_operator << <blocks, threads >> >(input, matrix_average, matrix_average, size);

  // We take it to the power of 2
  kernel_multiply_frames_float << <blocks, threads >> >(matrix_average, matrix_average, matrix_average, size);

  // And we take the average
  const float global_variance = average_operator(matrix_average, size);

  cudaFree(matrix_average);

  return global_variance;
}

static __global__ void kernel_float_to_complex(
  const float* input,
  cufftComplex* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index].x = input[index];
    output[index].y = input[index];
    index += blockDim.x * gridDim.x;
  }
}

static float average_local_variance(
  const float* input,
  const unsigned int square_size)
{
  unsigned int size = square_size * square_size;
  unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  /* ke matrix with same size than input */
  cufftComplex* ke_gpu_frame;
  size_t ke_gpu_frame_pitch;

  /* Allocates memory for ke_gpu_frame. */
  cudaMallocPitch(&ke_gpu_frame,
    &ke_gpu_frame_pitch,
    square_size * sizeof(cufftComplex),
    square_size);
  cudaMemset2D(
    ke_gpu_frame,
    ke_gpu_frame_pitch,
    0,
    square_size * sizeof(cufftComplex),
    square_size);

  {
    /* Build the 3x3 matrix */
    float ke_cpu[9] =
    {
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f
    };

    cufftComplex ke_complex_cpu[9];
    for (int i = 0; i < 9; ++i)
    {
      ke_complex_cpu[i].x = ke_cpu[i];
      ke_complex_cpu[i].y = ke_cpu[i];
    }

    /* Copy the ke matrix to ke_gpu_frame. */
    cudaMemcpy2D(
      ke_gpu_frame,
      ke_gpu_frame_pitch,
      ke_complex_cpu,
      3 * sizeof(cufftComplex),
      3 * sizeof(cufftComplex),
      3,
      cudaMemcpyHostToDevice);
  }

  cufftComplex* input_complex;
  cudaMalloc(&input_complex, size * sizeof(cufftComplex));

  /* Convert input float frame to complex frame. */
  kernel_float_to_complex << <blocks, threads >> >(input, input_complex, size);

  /* Allocation of convolution i * ke output */
  float* i_ke_convolution;
  cudaMalloc(&i_ke_convolution, size * sizeof(float));

  cufftHandle plan2d_x;
  cufftHandle plan2d_k;
  cufftPlan2d(&plan2d_x, square_size, square_size, CUFFT_C2C);
  cufftPlan2d(&plan2d_k, square_size, square_size, CUFFT_C2C);

  /* Compute i * ke. */
  convolution_operator(
    input_complex,
    ke_gpu_frame,
    i_ke_convolution,
    size,
    plan2d_x,
    plan2d_k);

  /* Compute i - i * ke. */
  kernel_minus_operator << <blocks, threads >> >(
    input,
    i_ke_convolution,
    i_ke_convolution,
    size);

  /* Compute (i - i * ke)^2 */
  kernel_multiply_frames_float << <blocks, threads >> >(
    i_ke_convolution,
    i_ke_convolution,
    i_ke_convolution,
    size);

  cudaDeviceSynchronize();

  const float average_local_variance = average_operator(i_ke_convolution, size);

  /* -- Free ressources -- */
  cufftDestroy(plan2d_x);
  cufftDestroy(plan2d_k);

  cudaFree(i_ke_convolution);

  cudaFree(input_complex);
  cudaFree(ke_gpu_frame);


  return average_local_variance;
}

static __global__ void kernel_plus_operator(
  const float* input_left,
  const float* input_right,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input_left[index] + input_right[index];
    index += blockDim.x * gridDim.x;
  }
}

static __global__ void kernel_sqrt_operator(
  const float* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = sqrtf(input[index]);
    index += blockDim.x * gridDim.x;
  }
}

static float sobel_operator(
  const float* input,
  unsigned int square_size)
{
  unsigned int size = square_size * square_size;
  unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  /* ks matrix with same size than input */
  cufftComplex* ks_gpu_frame;
  size_t ks_gpu_frame_pitch;

  /* Allocates memory for ks_gpu_frame. */
  cudaMallocPitch(&ks_gpu_frame,
    &ks_gpu_frame_pitch,
    square_size * sizeof(cufftComplex),
    square_size);
  cudaMemset2D(
    ks_gpu_frame,
    ks_gpu_frame_pitch,
    0,
    square_size * sizeof(cufftComplex),
    square_size);

  {
    /* Build the ks 3x3 matrix */
    float ks_cpu[9] =
    {
      1.0f, 0.0f, -1.0f,
      2.0f, 0.0f, -2.0f,
      1.0f, 0.0f, -1.0f
    };

    cufftComplex ks_complex_cpu[9];
    for (int i = 0; i < 9; ++i)
    {
      ks_complex_cpu[i].x = ks_cpu[i];
      ks_complex_cpu[i].y = ks_cpu[i];
    }

    /* Copy the ks matrix to ks_gpu_frame. */
    cudaMemcpy2D(
      ks_gpu_frame,
      ks_gpu_frame_pitch,
      ks_complex_cpu,
      3 * sizeof(cufftComplex),
      3 * sizeof(cufftComplex),
      3,
      cudaMemcpyHostToDevice);
  }

  /* kst matrix with same size than input */
  cufftComplex* kst_gpu_frame;
  size_t kst_gpu_frame_pitch;

  /* Allocates memory for kst_gpu_frame. */
  cudaMallocPitch(&kst_gpu_frame,
    &kst_gpu_frame_pitch,
    square_size * sizeof(cufftComplex),
    square_size);
  cudaMemset2D(
    kst_gpu_frame,
    kst_gpu_frame_pitch,
    0,
    square_size * sizeof(cufftComplex),
    square_size);

  {
    /* Build the kst 3x3 matrix */
    float kst_cpu[9] =
    {
      1.0f, 2.0f, 1.0f,
      0.0f, 0.0f, 0.0f,
      -1.0f, -2.0f, -1.0f
    };

    cufftComplex kst_complex_cpu[9];
    for (int i = 0; i < 9; ++i)
    {
      kst_complex_cpu[i].x = kst_cpu[i];
      kst_complex_cpu[i].y = kst_cpu[i];
    }

    /* Copy the kst matrix to kst_gpu_frame. */
    cudaMemcpy2D(
      kst_gpu_frame,
      kst_gpu_frame_pitch,
      kst_complex_cpu,
      3 * sizeof(cufftComplex),
      3 * sizeof(cufftComplex),
      3,
      cudaMemcpyHostToDevice);
  }

  cufftComplex* input_complex;
  cudaMalloc(&input_complex, size * sizeof(cufftComplex));

  /* Convert input float frame to complex frame. */
  kernel_float_to_complex << <blocks, threads >> >(input, input_complex, size);

  /* Allocation of convolution i * ks output */
  float* i_ks_convolution;
  cudaMalloc(&i_ks_convolution, size * sizeof(float));

  /* Allocation of convolution i * kst output */
  float* i_kst_convolution;
  cudaMalloc(&i_kst_convolution, size * sizeof(float));

  cufftHandle plan2d_x;
  cufftHandle plan2d_k;
  cufftPlan2d(&plan2d_x, square_size, square_size, CUFFT_C2C);
  cufftPlan2d(&plan2d_k, square_size, square_size, CUFFT_C2C);

  /* Compute i * ks. */
  convolution_operator(
    input_complex,
    ks_gpu_frame,
    i_ks_convolution,
    size,
    plan2d_x,
    plan2d_k);

  /* Compute (i * ks)^2 */
  kernel_multiply_frames_float << <blocks, threads >> >(
    i_ks_convolution,
    i_ks_convolution,
    i_ks_convolution,
    size);

  /* Compute i * kst. */
  convolution_operator(
    input_complex,
    kst_gpu_frame,
    i_kst_convolution,
    size,
    plan2d_x,
    plan2d_k);

  /* Compute (i * kst)^2 */
  kernel_multiply_frames_float << <blocks, threads >> >(
    i_kst_convolution,
    i_kst_convolution,
    i_kst_convolution,
    size);

  /* Compute (i * ks)^2 - (i * kst)^2 */
  kernel_plus_operator << <blocks, threads >> >(
    i_ks_convolution,
    i_kst_convolution,
    i_ks_convolution,
    size);

  kernel_sqrt_operator << <blocks, threads >> >(
    i_ks_convolution,
    i_ks_convolution,
    size);

  cudaDeviceSynchronize();

  const float average_magnitude = average_operator(i_ks_convolution, size);

  /* -- Free ressources -- */
  cufftDestroy(plan2d_x);
  cufftDestroy(plan2d_k);

  cudaFree(i_ks_convolution);
  cudaFree(i_kst_convolution);

  cudaFree(input_complex);

  cudaFree(kst_gpu_frame);
  cudaFree(ks_gpu_frame);

  // HEHEHEHEHEHEHEHEH
  return 1.0f / average_magnitude;
}



float focus_metric(
  float* input,
  unsigned int square_size)
{
  unsigned int size = square_size * square_size;
  unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  /* Divide each pixels to avoid higher values than float can contains. */
  kernel_float_divide << <blocks, threads >> >(input, size, static_cast<float>(size));

  const float global_variance = global_variance_intensity(input, size);
  const float avr_local_variance = average_local_variance(input, square_size);
  const float avr_magnitude = sobel_operator(input, square_size);

  return global_variance * avr_local_variance * avr_magnitude;
}
