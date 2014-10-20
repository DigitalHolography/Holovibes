#include "contrast_correction.cuh"
#ifndef __CUDACC__  
#define __CUDACC__
#endif


__global__ void make_histo(int *histo, unsigned char *img, int img_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < img_size)
  {
    atomicAdd(&histo[img[index]], 1);
  }
}

__global__ void sum_histok(int *histo, int *summed_histo)
{
  for (int i = 0; i < 256; i++)
  {
    for (int j = 0; j <= i; j++)
      summed_histo[i] += histo[j];
  }
}

__global__ void apply_correction(int *sum_histo, unsigned char *img, int img_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < img_size)
  {
    img[index] = ((256 - 1) * sum_histo[img[index]] / img_size);;
    index += blockDim.x * gridDim.x;
  }
}

void sum_histo_c(int *histo, int *summed_histo)
{
  /*for (int i = 0; i < 256; i++)
  {
    for (int j = 0; j <= i; j++)
      summed_histo[i] += histo[j];
  }*/

  summed_histo[0] = histo[0];
  for (int i = 1; i < 256; i++)
  {
      summed_histo[i] += summed_histo[i - 1] + histo[i];
  }

}

void correct_contrast(unsigned char *img,int img_size)
{
  int threads = 512;
  int blocks = (img_size + 511) / 512;

  int *histo;
  int *sum_histo;
  int *histo_cpu = (int*)calloc(sizeof(int) * 256,1);
  int *sum_histo_cpu = (int*)calloc(1,sizeof(int)* 256);
  cudaMalloc(&sum_histo, 256 * sizeof (int));
  cudaMalloc(&histo, 256 * sizeof (int));
  cudaMemset(histo, 0, 256 * sizeof(int));
  make_histo<<<blocks, threads >>>(histo, img, img_size);
  cudaMemcpy(histo_cpu, histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);
  sum_histo_c(histo_cpu, sum_histo_cpu);
  cudaMemcpy(sum_histo, sum_histo_cpu, 256 * sizeof(int), cudaMemcpyHostToDevice);
  /*for (int i = 0; i < 256; i++)
  {
    std::cout << histo_cpu[i] <<"--" << sum_histo_cpu[i] << std::endl;
  }*/
  apply_correction <<<blocks, threads >> >(sum_histo, img, img_size);
  cudaFree(histo);
  cudaFree(sum_histo);
  free(histo_cpu);
  free(sum_histo_cpu);
}