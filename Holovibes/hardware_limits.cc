#include "hardware_limits.hh"

unsigned int get_max_threads_1d()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.maxThreadsPerBlock;
}

unsigned int get_max_threads_2d()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return sqrt(prop.maxThreadsPerBlock);
}

unsigned int get_max_blocks()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.maxGridSize[0];
}