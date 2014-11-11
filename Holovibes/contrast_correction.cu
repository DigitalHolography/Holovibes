#include "contrast_correction.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdlib>

#include "hardware_limits.hh"

__global__ static void make_histo(int *histo, unsigned char *img, int img_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < img_size)
  {
    atomicAdd(&histo[img[index]], 1);
  }
}

__global__ static void apply_correction(int *sum_histo, unsigned char *img, int img_size, int tons)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < img_size)
  {
    img[index] = ((tons - 1) * sum_histo[img[index]] / img_size);;
    index += blockDim.x * gridDim.x;
  }
}

void sum_histo_c(int *histo, int *summed_histo, int bytedepth)
{
  int tons = 65536;
  if (bytedepth == 1)
    tons = 256;
  summed_histo[0] = histo[0];
  for (int i = 1; i < tons; i++)
  {
    summed_histo[i] += summed_histo[i - 1] + histo[i];
  }
}

void correct_contrast(unsigned char *img, int img_size, int bytedepth)
{
  int tons = 65536;
  if (bytedepth == 1)
    tons = 256;
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (img_size + threads - 1) / threads;
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  int *histo;
  int *sum_histo;
  int *histo_cpu = (int*)calloc(sizeof(int) * tons, 1);
  int *sum_histo_cpu = (int*)calloc(1, sizeof(int)* tons);
  cudaMalloc(&sum_histo, tons * sizeof(int));
  cudaMalloc(&histo, tons * sizeof(int));
  cudaMemset(histo, 0, tons * sizeof(int));
  make_histo<<<blocks, threads>>>(histo, img, img_size);
  cudaMemcpy(histo_cpu, histo, tons * sizeof(int), cudaMemcpyDeviceToHost);
  sum_histo_c(histo_cpu, sum_histo_cpu, bytedepth);
  cudaMemcpy(sum_histo, sum_histo_cpu, tons * sizeof(int), cudaMemcpyHostToDevice);
  apply_correction <<<blocks, threads>>>(sum_histo, img, img_size, tons);
  cudaFree(histo);
  cudaFree(sum_histo);
  free(histo_cpu);
  free(sum_histo_cpu);
}