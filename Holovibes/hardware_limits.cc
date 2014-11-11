#include "hardware_limits.hh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

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
  return static_cast<unsigned int>(sqrt(prop.maxThreadsPerBlock));
}

unsigned int get_max_blocks()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.maxGridSize[0];
}