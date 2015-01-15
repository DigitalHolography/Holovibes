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
  unsigned int size,
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

/*! \brief Find the minimum pixel value of an image and the maximum one.
*
* \param input The image to correct correct contrast.
* This image should be stored in device VideoRAM.
* \param size Size of the image in number of pixels.
* \param min Minimum pixel value found.
* \param max Maximum pixel value found.
*
*/
void auto_contrast_correction(
  float* input,
  unsigned int size,
  float* min,
  float* max)
{
  float* frame_cpu = new float[size]();
  cudaMemcpy(frame_cpu, input, sizeof(float) * size, cudaMemcpyDeviceToHost);
  find_min_max_img(frame_cpu, size, min, max);
  delete[] frame_cpu;

  if (*min < 1.0f)
    *min = 1.0f;
  if (*max < 1.0f)
    *max = 1.0f;
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

/*! \brief Make the contrast of the image depending of the
* maximum an minimum input given by the user.
*
* The algortihm used is a contrast stretching, the
* values min and max can be found thanks to the previous functions
* or can be setted by the user in case of a particular use.
* in case of autocontrast the same function is used but with found values on
* the image.
* This image should be stored in device VideoRAM.
* \param input The image to correct correct contrast.
* \param size Size of the image in number of pixels.
* \param min Minimum pixel value of the input image.
* \param max Maximum pixel value of the input image.
*
*/
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
