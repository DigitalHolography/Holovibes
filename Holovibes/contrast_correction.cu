#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <cstdlib>
#include <iostream>

#include "hardware_limits.hh"

#if 0
static __global__ void kernel_histogram(
  float* input,
  unsigned int input_size,
  unsigned int* histogram,
  unsigned int histogram_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    unsigned int pixel_value = __float2_uint_rz(input[index]);

    if (pixel_value >= histogram_size)
      pixel_value = histogram_size - 1;

    atomicAdd(histogram[pixel_value], 1);

    index += blockDim.x * gridDim.x;
  }
}
#endif

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
    for (unsigned int i = 0; i < 255; i++)
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
    for (unsigned int i = 0; i < 65535; i++)
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

void find_min_max_img(
  float *img_cpu,
  float *min,
  float *max,
  unsigned int nbpixels)
{
    *min = FLT_MAX;
    *max = FLT_MIN;
    for (int i = 0; i < nbpixels; i++)
    {
      if (img_cpu[i] > *max)
          *max = i;
        if (img_cpu[i] < *min)
          *min = i;
    }
}

void auto_contrast_correction(
  float* input,
  unsigned int size,
  float* min,
  float* max)
{
  float *img_cpu = (float *)malloc(size * sizeof (float));
  cudaMemcpy(img_cpu, input, sizeof(float)* size, cudaMemcpyDeviceToHost);
  find_min_max_img(input, min, max, size);
  free(img_cpu);
  std::cout << "min: " << *min << "max: " << *max << std::endl;
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
