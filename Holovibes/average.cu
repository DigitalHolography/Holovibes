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
  unsigned int size_x_s,
  unsigned int size_y_s,
  unsigned int size_x_n,
  unsigned int size_y_n,
  float *out_value,
  const camera::FrameDescriptor fd,
  unsigned int start_x_s,
  unsigned int start_y_s,
  unsigned int start_x_n,
  unsigned int start_y_n)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (fd.frame_res() + threads - 1) / threads;

  float s;
  float n;

  make_average << <blocks, threads >> >(image, size_x_n, size_y_n, &n, fd.frame_res(), start_x_n, start_y_n, fd);
  make_average << <blocks, threads >> >(image, size_x_s, size_y_s, &s, fd.frame_res(), start_x_s, start_y_s, fd);

  s = s * float(1 / float(size_x_s * size_y_s));
  n = n * float(1 / float(size_x_n * size_y_n));

  float moy = 10 * log10f(s / n);
  result_vect->push_back(moy);
}
