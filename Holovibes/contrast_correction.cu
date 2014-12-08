#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>

#include "hardware_limits.hh"

static void find_min_max_img(
  float *img_cpu,
  unsigned int size,
  float *min,
  float *max)
{
  *min = FLT_MAX;
  *max = FLT_MIN;
  for (unsigned int i = 0; i < size; i++)
  {
    if (img_cpu[i] > *max)
      *max = img_cpu[i];
    if (img_cpu[i] < *min)
      *min = img_cpu[i];
  }
}

void auto_contrast_correction(
  float* input,
  unsigned int size,
  float* min,
  float* max)
{
  float* frame_cpu = new float[size]();
  cudaMemcpy(frame_cpu, input, sizeof(float) * size, cudaMemcpyDeviceToHost);
  find_min_max_img(frame_cpu, size, min, max);
  delete[] frame_cpu;

  if (*min < 1.0f)
    *min = 1.0f;
  if (*max < 1.0f)
    *max = 1.0f;
}

static __global__ void apply_contrast(
  float* input,
  unsigned int size,
  float factor,
  float min)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    input[index] = factor * (input[index] - static_cast<float>(min));
    index += blockDim.x * gridDim.x;
  }
}

void manual_contrast_correction(
  float* input,
  unsigned int size,
  unsigned short dynamic_range,
  float min,
  float max)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  const float factor = static_cast<float>(dynamic_range) / (max - min);
  apply_contrast<<<blocks, threads>>>(input, size, factor, min);
}
