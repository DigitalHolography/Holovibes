#include "average.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "hardware_limits.hh"

/*! \brief  Sume 2 zone of input image
*
* \param input The image from where zones should be summed.
* \param width The width of the input image.
* \param height The height of the input image.
*
*/
static __global__ void kernel_zone_sum(
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

/*! \brief  Make the average plot on the 2 select zones
*
* \param input The image from where zones should be ploted
* \param width The width of the input image.
* \param height The height of the input image.
* \param signal Coordinates of the signal zone to use.
* \param noise Coordinates of the noise zone to use.
* \rerun A tupple of 3 floats <sum of signal zones pixels, sum of noise zone pixels, average>.
*
*/
std::tuple<float, float, float> make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise)
{
  unsigned int size = width * height;
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* gpu_s;
  float* gpu_n;

  cudaMalloc(&gpu_s, sizeof(float));
  cudaMalloc(&gpu_n, sizeof(float));

  cudaMemset(gpu_s, 0, sizeof(float));
  cudaMemset(gpu_n, 0, sizeof(float));

  unsigned int signal_width = abs(signal.top_right.x - signal.top_left.x);
  unsigned int signal_height = abs(signal.top_left.y - signal.bottom_left.y);
  unsigned int noise_width = abs(noise.top_right.x - noise.top_left.x);
  unsigned int noise_height = abs(noise.top_left.y - noise.bottom_left.y);

  kernel_zone_sum <<<blocks, threads>>>(input, width, height, gpu_n,
    noise.top_left.x, noise.top_left.y, noise_width, noise_height);
  kernel_zone_sum <<<blocks, threads>>>(input, width, height, gpu_s,
    signal.top_left.x, signal.top_left.y, signal_width, signal_height);

  float cpu_s;
  float cpu_n;

  cudaMemcpy(&cpu_s, gpu_s, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cpu_n, gpu_n, sizeof(float), cudaMemcpyDeviceToHost);

  cpu_s /=  float(signal_width * signal_height);
  cpu_n /=  float(noise_width * noise_height);

  float moy = 10 * log10f(cpu_s / cpu_n);

  cudaFree(gpu_n);
  cudaFree(gpu_s);

  return std::tuple<float, float, float>{ cpu_s, cpu_n, moy };
}
