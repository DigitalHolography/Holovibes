#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>
#include <algorithm>

#include "contrast_correction.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

static __global__ void apply_contrast(
  float* input,
  const unsigned int size,
  const float factor,
  const float min)
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
  const unsigned int size,
  const unsigned short dynamic_range,
  const float min,
  const float max,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  const float factor = static_cast<float>(dynamic_range) / (max - min);
  apply_contrast << <blocks, threads, 0, stream >> >(input, size, factor, min);
}

void auto_contrast_correction(
  float* input,
  const unsigned int size,
  float* min,
  float* max,
  cudaStream_t stream)
{
  float* frame_cpu = new float[size]();
  cudaMemcpyAsync(frame_cpu, input, sizeof(float)* size, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  auto minmax = std::minmax_element(frame_cpu, frame_cpu + size);
  *min = *minmax.first;
  *max = *minmax.second;

  delete[] frame_cpu;

  if (*min < 1.0f)
    *min = 1.0f;
  if (*max < 1.0f)
    *max = 1.0f;
}