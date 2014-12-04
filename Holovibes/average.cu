#include "average.cuh"

__global__ void make_average(float *image, unsigned int size_x, unsigned int size_y, float *out_value, unsigned int nb_pixels,
  unsigned int start_x, unsigned int start_y, const camera::FrameDescriptor fd)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < nb_pixels)
  {
    int x = index % fd.width;
    int y = index / fd.height;
    if (x >= start_x && x < start_x + size_x
      && y >= start_y && y < start_y + size_y)
    {
      atomicAdd(out_value, image[index]);
    }
  }
}

void make_average_plot(std::vector<float> *result_vect,
  float *image,
  const camera::FrameDescriptor fd,
  holovibes::Rectangle& signal,
  holovibes::Rectangle& noise)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (fd.frame_res() + threads - 1) / threads;

  float s;
  float n;

  unsigned int signal_width = abs(signal.top_right.x - signal.top_left.x);
  unsigned int signal_height = abs(signal.bottom_left.x - signal.bottom_right.x);
  unsigned int noise_width = abs(noise.top_right.x - noise.top_left.x);
  unsigned int noise_height = abs(noise.bottom_left.x - noise.bottom_right.x);

  make_average << <blocks, threads >> >(image, noise_width, noise_height, &n, fd.frame_res(), noise.top_left.x, noise.top_left.y, fd);
  make_average << <blocks, threads >> >(image, signal_width, signal_height, &s, fd.frame_res(), signal.top_left.x, signal.top_left.y, fd);

  s = s * float(1 / float(signal_width * signal_height));
  n = n * float(1 / float(noise_width * noise_height));

  float moy = 10 * log10f(s / n);
  result_vect->push_back(moy);
}
