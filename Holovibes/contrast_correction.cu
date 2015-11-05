#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>

#include "hardware_limits.hh"

/*! \brief  Find the minimum pixel value of an image and the maximum one.
*
* \param img_cpu The image to searche values in.
* This image should be stored in host RAM.
* \param size Size of the image in number of pixels.
* \param min Minimum pixel value found.
* \param max Maximum pixel value found.
*
*/
static void find_min_max_img(
  float *img_cpu,
  const unsigned int size,
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
  const float max)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  const float factor = static_cast<float>(dynamic_range) / (max - min);
  apply_contrast << <blocks, threads >> >(input, size, factor, min);
}

void auto_contrast_correction(
  float* input,
  const unsigned int size,
  float* min,
  float* max)
{
  float* frame_cpu = new float[size]();
  cudaMemcpy(frame_cpu, input, sizeof(float)* size, cudaMemcpyDeviceToHost);
  find_min_max_img(frame_cpu, size, min, max);
  delete[] frame_cpu;

  if (*min < 1.0f)
    *min = 1.0f;
  if (*max < 1.0f)
    *max = 1.0f;
}