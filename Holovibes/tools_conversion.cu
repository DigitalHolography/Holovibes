#include <algorithm>
#include <device_launch_parameters.h>

#include "tools_conversion.cuh"
#include "hardware_limits.hh"
#include "tools.hh"
#include <iostream>

__global__ void img8_to_complex(
  cufftComplex* output,
  const unsigned char* input,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    // Image rescaling on 2^16 colors (65535 / 255 = 257)
	  float val = static_cast<float>(input[index] * 257);
	output[index].x = val;
	output[index].y = 0;
    index += blockDim.x * gridDim.x;
  }
}


__global__ void img16_to_complex(
  cufftComplex* output,
  const unsigned short* input,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
	  float val = static_cast<float>(input[index]);
	  output[index].x = val;
	  output[index].y = 0;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void float_to_complex(
	cufftComplex* output,
	const float* input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		float val = input[index];
		output[index].x = val;
		output[index].y = 0;
		index += blockDim.x * gridDim.x;
	}
}

/* Kernel function wrapped by complex_to_modulus. */
static __global__ void kernel_complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  { 
	  output[index] = hypotf(input[index].x, input[index].y);
    index += blockDim.x * gridDim.x;
  }
}

void complex_to_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_complex_to_modulus << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Kernel function wrapped in complex_to_squared_modulus. */
static __global__ void kernel_complex_to_squared_modulus(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
	  output[index] = pow(hypotf(input[index].x, input[index].y), 2);
    index += blockDim.x * gridDim.x;
  }
}

void complex_to_squared_modulus(
  const  cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_complex_to_squared_modulus << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Kernel function wrapped in complex_to_argument. */
static __global__ void kernel_complex_to_argument(
  const cufftComplex* input,
  float* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const float pi_div_2 = M_PI / 2.0f;

  while (index < size)
  {
	  output[index] = (atanf(input[index].y / input[index].x) + pi_div_2);;

    index += blockDim.x * gridDim.x;
  }
}

void complex_to_argument(
  const cufftComplex* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_complex_to_argument << <blocks, threads, 0, stream >> >(input, output, size);
}

/* Find the minimum and the maximum of a floating-point array.
 *
 * The minimum and maximum can't be computed directly, because blocks
 * cannot communicate. Hence we compute local minima and maxima and
 * put them in two arrays.
 *
 * \param Size Number of threads in a block for this kernel.
 * Also, it's the size of min and max.
 * \param min Array of Size floats, which will contain local minima.
 * \param max Array of Size floats, which will contain local maxima.
 */
template <unsigned Size>
static __global__ void kernel_minmax(
  const float* data,
  const size_t size,
  float* min,
  float* max)
{
  __shared__ float local_min[Size];
  __shared__ float local_max[Size];

  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index > size)
    return;
  local_min[threadIdx.x] = data[index];
  local_max[threadIdx.x] = data[index];

  __syncthreads();

  if (threadIdx.x == 0)
  {
    /* Accumulate the results of the neighbors, computing min-max values,
     * and store them in the first element of local arrays. */
    for (auto i = 1; i < Size; ++i)
    {
      if (local_min[i] < local_min[0])
        local_min[0] = local_min[i];
      if (local_max[i] > local_max[0])
        local_max[0] = local_max[i];
    }
    min[blockIdx.x] = local_min[0];
    max[blockIdx.x] = local_max[0];
  }
}

template <typename T>
static __global__ void kernel_rescale(T* data,
  const size_t size,
  const T min,
  const T max,
  const T new_max)
{
  const unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index > size)
    return;

  data[index] = (data[index] + fabsf(min)) * new_max / (fabsf(max) + fabsf(min));
}

void rescale_float(
  const float* input,
  float* output,
  const unsigned int size,
  cudaStream_t stream)
{
  const unsigned threads = 128;
  unsigned blocks = map_blocks_to_problem(size, threads);

  // TODO : See if gpu_float_buffer_ could be used directly.
  cudaMemcpy(output, input, sizeof(float) * size, cudaMemcpyDeviceToDevice);

  // Computing minimum and maximum values, in order to rescale properly.
  float* gpu_local_mins;
  float* gpu_local_maxs;
  cudaMalloc(&gpu_local_mins, sizeof(float)* blocks);
  cudaMalloc(&gpu_local_maxs, sizeof(float)* blocks);

  /* We have to hardcode the template parameter, unfortunately.
   * It must be equal to the number of threads per block. */
  kernel_minmax <128> << <blocks, threads, threads * 2, stream >> > (output,
    size,
    gpu_local_mins,
    gpu_local_maxs);

  float* cpu_local_mins = new float[blocks];
  float* cpu_local_maxs = new float[blocks];
  cudaMemcpy(cpu_local_mins, gpu_local_mins, sizeof(float)* blocks, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_local_maxs, gpu_local_maxs, sizeof(float)* blocks, cudaMemcpyDeviceToHost);

  const float max_intensity = 65535.f;
  kernel_rescale << <blocks, threads, 0, stream >> >(
    output,
    size,
    *(std::min_element(cpu_local_mins, cpu_local_mins + threads)),
    *(std::max_element(cpu_local_maxs, cpu_local_maxs + threads)),
    max_intensity);

  cudaFree(gpu_local_mins);
  cudaFree(gpu_local_maxs);
}

void rescale_float_unwrap2d(
	float *input,
	float *output,
	float *cpu_buffer,
	unsigned int frame_res,
	cudaStream_t stream)
{
	float min = 0;
	float max = 0;
	const unsigned threads = 128;
	const unsigned blocks = map_blocks_to_problem(frame_res, threads);

	cudaMemcpy(cpu_buffer, input, sizeof(float) * frame_res, cudaMemcpyDeviceToHost);
	auto minmax = std::minmax_element(cpu_buffer, cpu_buffer + frame_res);
	min = *minmax.first;
	max = *minmax.second;

	cudaMemcpy(output, input, sizeof(float)* frame_res, cudaMemcpyDeviceToDevice);

	kernel_normalize_images << < blocks, threads, 0, stream >> > (
	output,
	max,
	min,
	frame_res);
}

__global__ void kernel_rescale_argument(
	float *input,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		input[index] *= 65535.0f / M_PI;

		index += blockDim.x * gridDim.x;
	}
}

void rescale_argument(
	float *input,
	const unsigned int frame_res,
	cudaStream_t stream)
{
	const unsigned int threads = get_max_threads_1d();
	const unsigned int blocks = map_blocks_to_problem(frame_res, threads);

	kernel_rescale_argument << <blocks, threads, 0, stream >> >(input, frame_res);
}

/*! \brief Kernel function wrapped in endianness_conversion, making
 ** the call easier
 **/
static __global__ void kernel_endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = (input[index] << 8) | (input[index] >> 8);

    index += blockDim.x * gridDim.x;
  }
}

void endianness_conversion(
  const unsigned short* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_endianness_conversion << <blocks, threads, 0, stream >> >(input, output, size);
}

/*! \brief Kernel function wrapped in float_to_ushort, making
 ** the call easier
 **/
static __global__ void kernel_float_to_ushort(
  const float* input,
  unsigned short* output,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    if (input[index] > 65535.0f)
      output[index] = 65535;
    else if (input[index] < 0.0f)
      output[index] = 0;
    else
      output[index] = static_cast<unsigned short>(input[index]);

    index += blockDim.x * gridDim.x;
  }
}

static __global__ void kernel_complex_to_ushort(
	const cufftComplex* input,
	unsigned int * output,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		unsigned short x = 0;
		unsigned short y = 0;
		if (input[index].x > 65535.0f)
			x = 65535;
		else if (input[index].x >= 1.0f)
			x = static_cast<unsigned short>(pow(input[index].x, 2));
		
		if (input[index].y > 65535.0f)
			y = 65535;
		else if (input[index].y >= 0.0f)
			y = static_cast<unsigned short>(pow(input[index].y, 2));
		auto& res = output[index];
		res ^= res;
		res = x << 16;
		res += y;
		index += blockDim.x * gridDim.x;
	}
}

void float_to_ushort(
  const float* input,
  unsigned short* output,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_float_to_ushort << <blocks, threads, 0, stream >> >(input, output, size);
}

void complex_to_ushort(
	const cufftComplex* input,
	unsigned int* output,
	const unsigned int size,
	cudaStream_t stream)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(size, threads);

	kernel_complex_to_ushort << <blocks, threads, 0 >> >(input, output, size);
}

/*! \brief Memcpy of a complex sized frame into another buffer */
void complex_to_complex(
	const cufftComplex* input,
	unsigned short* output,
	const unsigned int size,
	cudaStream_t stream)
{
	cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice);
}

__global__ void	kernel_buffer_size_conversion(char *real_buffer
	, const char *buffer
	, const size_t frame_desc_width
	, const size_t frame_desc_height
	, const size_t real_frame_desc_width
	, const size_t area)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < area)
	{
		unsigned int x = index % real_frame_desc_width;
		unsigned int y = index / real_frame_desc_width;
		if (y < frame_desc_height && x < frame_desc_width)
			real_buffer[index] = buffer[y * frame_desc_width + x];
		index += blockDim.x * gridDim.x;
	}
}

void	buffer_size_conversion(char *real_buffer
	, const char *buffer
	, const camera::FrameDescriptor real_frame_desc
	, const camera::FrameDescriptor frame_desc)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem((frame_desc.height * real_frame_desc.width * static_cast<size_t>(frame_desc.depth)), threads);

	kernel_buffer_size_conversion <<<blocks, threads, 0>>>(
		real_buffer,
		buffer,
		frame_desc.width * static_cast<unsigned int>(frame_desc.depth),
		frame_desc.height * static_cast<unsigned int>(frame_desc.depth),
		real_frame_desc.width * static_cast<unsigned int>(frame_desc.depth),
		frame_desc.height * real_frame_desc.width * static_cast<size_t>(frame_desc.depth));
}

__global__ void kernel_accumulate_images(
	const float *input,
	float *output,
	const size_t start,
	const size_t max_elmt,
	const size_t nb_elmt,
	const size_t nb_pixel)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t	i = 0;
	int		pos = start;

	if (index < nb_pixel)
	{
		output[index] = 0;
		while (i < nb_elmt)
		{
			output[index] += input[index + pos * nb_pixel];
			i++;
			pos--;
			if (pos < 0)
				pos = max_elmt - 1;
		}
		output[index] /= nb_elmt;
	}
}

/*! \brief Kernel function wrapped in accumulate_images, making
** the call easier
**/
void accumulate_images(
	const float *input,
	float *output,
	const size_t start,
	const size_t max_elmt,
	const size_t nb_elmt,
	const size_t nb_pixel,
	cudaStream_t stream)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(nb_pixel, threads);

	kernel_accumulate_images << <blocks, threads, 0, stream >> >(
		input,
		output,
		start,
		max_elmt,
		nb_elmt,
		nb_pixel);
}

__global__ void kernel_normalize_images(
	float *image,
	const float max,
	const float min,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		if (min < 0.f)
			image[index] = (image[index] + fabs(min)) / (fabs(min) + max) * 65535.0f;
		else
			image[index] = (image[index] - min) / (max - min) * 65535.0f;
		index += blockDim.x * gridDim.x;
	}
}