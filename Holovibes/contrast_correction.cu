#include <device_launch_parameters.h>
#include <float.h>
#include <algorithm>

#include "contrast_correction.cuh"
#include "hardware_limits.hh"
#include "tools.hh"

static __global__ void apply_contrast(	float		*input,
										const uint	size,
										const float	factor,
										const float	min)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    input[index] = factor * (input[index] - static_cast<float>(min));
    index += blockDim.x * gridDim.x;
  }
}

void manual_contrast_correction(float			*input,
								const uint		size,
								const ushort	dynamic_range,
								const float		min,
								const float		max,
								cudaStream_t	stream)
{
  uint threads = get_max_threads_1d();
  uint blocks = map_blocks_to_problem(size, threads);

  const float factor = static_cast<float>(dynamic_range) / (max - min);
  apply_contrast << <blocks, threads, 0, stream >> >(input, size, factor, min);
}

void auto_contrast_correction(	float			*input,
								const uint		size,
								float			*min,
								float			*max,
								cudaStream_t	stream)
{
  float	*frame_cpu = new float[size]();
  cudaMemcpyAsync(frame_cpu, input, sizeof(float)* size, cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  auto minmax = std::minmax_element(frame_cpu, frame_cpu + size);
  *min = *minmax.first;
  *max = *minmax.second;

  delete[] frame_cpu;

  *min = ((*min < 1.0f) ? (1.0f) : (*min));
  *max = ((*max < 1.0f) ? (1.0f) : (*max));
}