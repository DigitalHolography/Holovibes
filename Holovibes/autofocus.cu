#include "autofocus.cuh"

# include <cuda_runtime.h>
# include "device_launch_parameters.h"
# include "hardware_limits.hh"
# include "tools.cuh"
# include "average.cuh"

/* -- REMOVE THIS -- */
# include <iostream>

static float global_variance_intensity(
  const float* input,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* squared_input;
  cudaMalloc<float>(&squared_input, size * sizeof(float));

  kernel_multiply_frames_float <<<blocks, threads>>>(input, input, squared_input, size);

  float average_squared_input = average_operator(squared_input, size);
  float average_input = average_operator(input, size);
  float average_input_squared = average_input * average_input;

  float global_variance = average_squared_input - average_input_squared;

  cudaFree(squared_input);

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
    cufftComplex ke_19;
    ke_19.x = sqrtf(1.0f / 9.0f);
    ke_19.y = sqrtf(1.0f / 9.0f);

    /* Build the ke 3x3 matrix */
    cufftComplex ke_cpu[9];
    for (int i = 0; i < 9; ++i)
      ke_cpu[i] = ke_19;

    /* Copy the ke matrix to ke_gpu_frame. */
    cudaMemcpy2D(
      ke_gpu_frame,
      ke_gpu_frame_pitch,
      ke_cpu,
      3 * sizeof(cufftComplex),
      3 * sizeof(cufftComplex),
      3,
      cudaMemcpyHostToDevice);
  }

  cufftComplex* input_complex;
  cudaMalloc(&input_complex, size * sizeof(cufftComplex));

  /* Convert input float frame to complex frame. */
  kernel_float_to_complex <<<blocks, threads>>>(input, input_complex, size);

  /* Allocation of convolution i * ke output */
  float* i_ke_convolution;
  cudaMalloc(&i_ke_convolution, size * sizeof(float));

  /* Allocation of convolution i^2 * ke output */
  float* i2_ke_convolution;
  cudaMalloc(&i2_ke_convolution, size * sizeof(float));

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

  /* Compute (i * ke)^2 */
  kernel_multiply_frames_float<<<blocks, threads>>>(
    i_ke_convolution,
    i_ke_convolution,
    i_ke_convolution,
    size);

  /* Compute i^2 * ke. */
  kernel_multiply_frames_complex<<<blocks, threads>>>(
    input_complex,
    input_complex,
    input_complex,
    size);

  convolution_operator(
    input_complex,
    ke_gpu_frame,
    i2_ke_convolution,
    size,
    plan2d_x,
    plan2d_k);

  /* Compute (i^2 * ke) - (i * ke)^2 */
  kernel_minus_operator<<<blocks, threads>>>(
    i2_ke_convolution,
    i_ke_convolution,
    i_ke_convolution,
    size);

  cudaDeviceSynchronize();

  float average_local_variance = average_operator(i_ke_convolution, size);

  /* -- Free ressources -- */
  cufftDestroy(plan2d_x);
  cufftDestroy(plan2d_k);

  cudaFree(i2_ke_convolution);
  cudaFree(i_ke_convolution);

  cudaFree(input_complex);
  cudaFree(ke_gpu_frame);

  return average_local_variance;
}

/* TODO: Fix me */
float focus_metric(
  float* input,
  unsigned int square_size)
{
  unsigned int size = square_size * square_size;
  float global_variance = global_variance_intensity(input, size);
  float avr_local_variance = average_local_variance(input, square_size);

  std::cout << "global variance: " << global_variance << std::endl;
  std::cout << "avr_local_variance: " << avr_local_variance << std::endl;

  return global_variance * avr_local_variance;
}
