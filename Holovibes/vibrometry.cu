#include <device_launch_parameters.h>
#include <cfloat>
#include <cuda_runtime.h>

#include "vibrometry.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

static __global__ void kernel_frame_ratio(
  const cufftComplex* frame_p,
  const cufftComplex* frame_q,
  cufftComplex* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    /* frame_p: a + ib */
    const float a = frame_p[index].x;
    const float b = frame_p[index].y;

    /* frame_q: c + id */
    const float c = frame_q[index].x;
    const float d = frame_q[index].y;

    const float q_squared_modulus = c * c + d * d + FLT_EPSILON;

    output[index].x = (a * c + b * d) / q_squared_modulus;
    output[index].y = (b * c - a * d) / q_squared_modulus;

    index += blockDim.x * gridDim.x;
  }
}

void frame_ratio(
  const cufftComplex* frame_p,
  const cufftComplex* frame_q,
  cufftComplex* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_frame_ratio << <blocks, threads, 0, stream >> >(frame_p, frame_q, output, size);
}