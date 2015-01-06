#include "average.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "hardware_limits.hh"

static __global__ void kernel_sum(
  float* input,
  unsigned int width,
  unsigned int height,
  float* output,
  unsigned int zone_start_x,
  unsigned int zone_start_y,
  unsigned int zone_width,
  unsigned int zone_height)
{
  unsigned int size = width * height;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    int x = index % width;
    int y = index / height;

    if (x >= zone_start_x && x < zone_start_x + zone_width
      && y >= zone_start_y && y < zone_start_y + zone_height)
    {
      atomicAdd(output, input[index]);
    }

    index += blockDim.x * gridDim.x;
  }
}

/* -- AVERAGE OPERATOR -- */
float average_operator(
  float* input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& zone)
{
  const unsigned int size = width * height;
  const unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* gpu_sum;
  cudaMalloc<float>(&gpu_sum, sizeof(float));
  cudaMemset(gpu_sum, 0, sizeof(float));

  const unsigned int zone_width = abs(zone.top_right.x - zone.top_left.x);
  const unsigned int zone_height = abs(zone.bottom_left.y - zone.top_left.y);

  kernel_sum <<<blocks, threads >>>(
    input,
    width,
    height,
    gpu_sum,
    zone.top_left.x,
    zone.top_left.y,
    zone_width,
    zone_height);

  float cpu_sum;
  cudaMemcpy(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpu_sum);

  cpu_sum /= float(zone_width * zone_height);

  return cpu_sum;
}

/* -- VIBROMETRY -- */
std::tuple<float, float, float> make_average_plot(
  float* input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise)
{
  float cpu_s = average_operator(input, width, height, signal);
  float cpu_n = average_operator(input, width, height, noise);

  float moy = 10 * log10f(cpu_s / cpu_n);

  return std::tuple<float, float, float>{ cpu_s, cpu_n, moy };
}
