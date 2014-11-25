#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

#include "hardware_limits.hh"

static __global__ void make_histo(
  int *histo,
  void *img,
  int img_size,
  int bytedepth)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < img_size)
  {
    if (bytedepth == 1)
      atomicAdd(&histo[((unsigned char*)img)[index]], 1);
    else
      atomicAdd(&histo[((unsigned short*)img)[index]], 1);
  }
}

static void find_min_max(
  unsigned int *min,
  unsigned int *max,
  int *histo,
  int bytedepth,
  int percent,
  unsigned int nbpixels)
{
  int acceptable = (percent / 100) * nbpixels;
  if (bytedepth == 1)
  {
    *min = 255;
    *max = 0;
    for (int i = 0; i < 255; i++)
    {
      if (histo[i] > acceptable)
      {
        if (i > *max)
          *max = i;
        if (i < *min)
          *min = i;
      }
    }
  }
  else
  {
    *min = 65535;
    *max = 0;
    for (int i = 0; i < 65535; i++)
    {
      if (histo[i] > acceptable)
      {
        if (i > *max)
          *max = i;
        if (i < *min)
          *min = i;
      }
    }
  }
}

// Fix this
#if 0
void auto_contrast_correction(
  float* input,
  unsigned int size,
  unsigned int* min,
  unsigned int* max,
  float threshold) // percent
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  int *histo;
  int *histo_cpu = (int*)calloc(sizeof(int)* tons, 1);
  cudaMalloc(&histo, tons * sizeof(int));
  cudaMemset(histo, 0, tons * sizeof(int));
  make_histo << <blocks, threads >> >(histo, img, img_size, bytedepth);
  cudaMemcpy(histo_cpu, histo, tons * sizeof(int), cudaMemcpyDeviceToHost);
  find_min_max(min, max, histo_cpu, bytedepth, percent, img_size);
  float factor = tons / (*max - *min);
  cudaFree(histo);
  free(histo_cpu);
}
#endif

static __global__ void apply_contrast(
  float* input,
  unsigned int size,
  float factor,
  unsigned short min)
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
  unsigned short min,
  unsigned short max)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  const float factor = static_cast<float>(dynamic_range) / static_cast<float>(max - min);
  apply_contrast<<<blocks, threads>>>(input, size, factor, min);
}
