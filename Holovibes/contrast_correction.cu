#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

#include "hardware_limits.hh"

__global__ void make_histo(
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

__global__ void apply_contrast(
  unsigned int min,
  float factor,
  void *img,
  unsigned int size,
  int bytedepth)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    if (bytedepth == 1)
    {
      int value = factor * (float)(((unsigned char*)img)[index] - min);
      if (value > 255)
        value = 255;
      else if (value < 0)
        value = 0;
      ((unsigned char*)img)[index] = value;
    }
    else
    {
      int value = factor * (float)(((unsigned short*)img)[index] - min);
      if (value > 65535)
        value = 65535;
      else if (value < 0)
        value = 0;
      ((unsigned short*)img)[index] = value;
    }
    index += blockDim.x * gridDim.x;
  }
}

void find_min_max(
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

void manual_contrast_correction(
  void *img,
  unsigned int img_size,
  int bytedepth,
  unsigned int manual_min,
  unsigned int manual_max)
{
  int tons = 65536;
  if (bytedepth == 1)
    tons = 256;
  int threads = get_max_threads_1d();
  int blocks = (img_size + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;
  float factor = tons / (manual_max - manual_min);
  apply_contrast <<<blocks, threads>>>(manual_min, factor, img, img_size, bytedepth);
}

void auto_contrast_correction(
  unsigned int *min,
  unsigned int *max,
  void *img,
  unsigned int img_size,
  unsigned int bytedepth,
  unsigned int percent)
{
  int tons = 65536;
  if (bytedepth == 1)
    tons = 256;
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (img_size + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;
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
