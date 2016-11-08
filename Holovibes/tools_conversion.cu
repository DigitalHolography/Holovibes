#include <algorithm>
#include <device_launch_parameters.h>

#include "tools_conversion.cuh"
#include "hardware_limits.hh"
#include "tools.hh"
#include <iostream>

__global__ void img8_to_complex(
  cufftComplex* output,
  const unsigned char* input,
  const unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    // Image rescaling on 2^16 colors (65535 / 255 = 257)
    float val  = sqrt(static_cast<float>(2 * (input[index] * 257)));
	output[index].x = val;
	output[index].y = val;
    index += blockDim.x * gridDim.x;
  }
}

//TODO:   removed the sqrt_array from computation and it seems more effective without.
//        need to do proper benchmarks.

__global__ void img16_to_complex(
  cufftComplex* output,
  const unsigned short* input,
  const unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
	  float val = sqrt(2 * static_cast<float>(input[index]));
	  output[index].x = val;
	  output[index].y = val;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void float_to_complex(
	cufftComplex* output,
	const float* input,
	const unsigned int size,
	const float* sqrt_array)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	while (index < size)
	{
		float val = input[index];
		if (val < 0)
		{
			val = abs(val);
			val = sqrtf(val);
			val *= -1;
		}
		else
			val = sqrtf(val);
		output[index].x = val;
		output[index].y = val;
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
	  output[index] = pow(hypotf(input[index].x, input[index].y), 2);//input[index].x * input[index].x + input[index].y * input[index].y;
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
  const float c = 65535.0f / M_PI;

  while (index < size)
  {
    output[index] = (atanf(input[index].y / input[index].x) + pi_div_2) * c;

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
  cudaMemcpy(output, input, sizeof(float)* size, cudaMemcpyDeviceToDevice);

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
		res ^ res;
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

__global__ void kernel_accumulate_images(
	const cufftComplex *input,
	cufftComplex *output,
	const size_t start,
	const size_t max_elmt,
	const size_t nb_elmt,
	const size_t nb_pixel)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	size_t	i = 0;
	size_t	pos = start;

	if (index < nb_pixel)
	{
		output[index].x = 0;
		output[index].y = 0;
		while (i < nb_elmt)
		{
			output[index].x += input[index + pos * nb_pixel].x;
			output[index].y += input[index + pos * nb_pixel].y;
			i++;
			pos++;
			if (pos > max_elmt)
				pos = 0;
		}
	}
}

void accumulate_images(
	const cufftComplex *input,
	cufftComplex *output,
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