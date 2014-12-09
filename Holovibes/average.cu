#include "average.cuh"

static __global__ void kernel_sum(
  float* input,
  unsigned int width,
  unsigned int height,
  float* output,
  unsigned int z_start_x,
  unsigned int z_start_y,
  unsigned int z_width,
  unsigned int z_height)
{
  unsigned int size = width * height;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    int x = index % width;
    int y = index / height;

    if (x >= z_start_x && x < z_start_x + z_width
      && y >= z_start_y && y < z_start_y + z_height)
    {
      atomicAdd(output, input[index]);
    }

    index += blockDim.x * gridDim.x;
  }
}

void make_average_plot(
  float *input,
  const unsigned int width,
  const unsigned int height,
  std::vector<std::tuple<float, float, float>>& output,
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

  unsigned int signal_width = abs(signal.top_right.x - signal.top_left.x);
  unsigned int signal_height = abs(signal.top_left.y - signal.bottom_left.y);
  unsigned int noise_width = abs(noise.top_right.x - noise.top_left.x);
  unsigned int noise_height = abs(noise.top_left.y - noise.bottom_left.y);

  kernel_sum <<<blocks, threads>>>(input, width, height, gpu_n,
    noise.top_left.x, noise.top_left.y, noise_width, noise_height);
  kernel_sum <<<blocks, threads>>>(input, width, height, gpu_s,
    signal.top_left.x, signal.top_left.y, signal_width, signal_height);

  float cpu_s;
  float cpu_n;

  cudaMemcpy(&cpu_s, gpu_s, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cpu_n, gpu_n, sizeof(float), cudaMemcpyDeviceToHost);

  cpu_s /=  float(signal_width * signal_height);
  cpu_n /=  float(noise_width * noise_height);

  float moy = 10 * log10f(cpu_s / cpu_n);

  output.push_back(std::tuple < float, float, float > { cpu_s, cpu_n, moy });

  cudaFree(gpu_n);
  cudaFree(gpu_s);
}
