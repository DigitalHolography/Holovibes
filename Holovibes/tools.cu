#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_unwrap.cuh"
#include "tools.hh"
#include "geometry.hh"
#include "hardware_limits.hh"
#include "compute_bundles.hh"

__global__ void kernel_apply_lens(
  cufftComplex *input,
  const unsigned int input_size,
  const cufftComplex *lens,
  const unsigned int lens_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    unsigned int index2 = index % lens_size; // necessary if more than one frame
	float tmp_x = input[index].x;
	input[index].x = input[index].x * lens[index2].x - input[index].y * lens[index2].y;
	input[index].y = input[index].y * lens[index2].x + tmp_x * lens[index2].y;
    index += blockDim.x * gridDim.x;
  }
}

static __global__ void kernel_shift_corners(
  float* input,
  const unsigned int size_x,
  const unsigned int size_y)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int index = j * blockDim.x * gridDim.x + i;
  unsigned int ni = 0;
  unsigned int nj = 0;
  unsigned int nindex = 0;
  float tmp = 0.0f;

  // Superior half of the matrix
  if (j >= size_y / 2)
  {
    // Left superior quarter of the matrix
    if (i < size_x / 2)
    {
      ni = i + size_x / 2;
      nj = j - size_y / 2;
    }
    // Right superior quarter
    else
    {
      ni = i - size_x / 2;
      nj = j - size_y / 2;
    }

    nindex = nj * size_x + ni;

    tmp = input[nindex];
    input[nindex] = input[index];
    input[index] = tmp;
  }
}

void shift_corners(
  float* input,
  const unsigned int size_x,
  const unsigned int size_y,
  cudaStream_t stream)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

  kernel_shift_corners << < lblocks, lthreads, 0, stream >> >(input, size_x, size_y);
}


/* Kernel used in apply_log10 */
static __global__ void kernel_log10(
  float* input,
  const unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    input[index] = log10f(input[index]);

    index += blockDim.x * gridDim.x;
  }
}

void apply_log10(
  float* input,
  const unsigned int size,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  kernel_log10 << <blocks, threads, 0, stream >> >(input, size);
}

/* Kernel used in convolution_operator */
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

void demodulation(
	cufftComplex* input,
	const cufftHandle  plan1d,
	cudaStream_t stream)
{
	// FFT 1D TEMPORAL
	cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);
}


void convolution_operator(
  const cufftComplex* x,
  const cufftComplex* k,
  float* out,
  const unsigned int size,
  const cufftHandle plan2d_x,
  const cufftHandle plan2d_k,
  cudaStream_t stream)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(size, threads);

  /* The convolution operator is used only when using autofocus feature.
   * It could be optimized but it's useless since it will be used occasionnally. */
  cufftComplex* tmp_x;
  cufftComplex* tmp_k;
  cudaMalloc<cufftComplex>(&tmp_x, size * sizeof(cufftComplex));
  cudaMalloc<cufftComplex>(&tmp_k, size * sizeof(cufftComplex));

  cufftExecC2C(plan2d_x, const_cast<cufftComplex*>(x), tmp_x, CUFFT_FORWARD);
  cufftExecC2C(plan2d_k, const_cast<cufftComplex*>(k), tmp_k, CUFFT_FORWARD);

  cudaStreamSynchronize(stream);

  kernel_multiply_frames_complex << <blocks, threads, 0, stream >> >(tmp_x, tmp_k, tmp_x, size);

  cudaStreamSynchronize(stream);

  cufftExecC2C(plan2d_x, tmp_x, tmp_x, CUFFT_INVERSE);

  cudaStreamSynchronize(stream);

  kernel_complex_to_modulus << <blocks, threads, 0, stream >> >(tmp_x, out, size);

  cudaFree(tmp_x);
  cudaFree(tmp_k);
}

void frame_memcpy(
  float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width,
  cudaStream_t stream)
{
  const float* zone_ptr = input + (zone.top_left.y * input_width + zone.top_left.x);

  cudaMemcpy2DAsync(
    output,
    output_width * sizeof(float),
    zone_ptr,
    input_width * sizeof(float),
    output_width * sizeof(float),
    output_width,
    cudaMemcpyDeviceToDevice,
    stream);
  cudaStreamSynchronize(stream);
}

/* Kernel helper used in average.
 *
 * Sums up the *size* first elements of input and stores the result in sum.
 *
 * SpanSize is the number of elements processed by a single thread.
 * This way of doing things comes from the empirical fact that (at the point
 * of this writing) loop unrolling in CUDA kernels may prove more efficient,
 * when the operation is really small. */
template <unsigned SpanSize>
static __global__ void kernel_sum(const float* input, float* sum, const size_t size)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index + SpanSize - 1) < size && (index % SpanSize) == 0)
  {
    float tmp_reduce = 0.0f;
    for (unsigned i = 0; i < SpanSize; ++i)
      tmp_reduce += input[index + i];
    // Atomic operation is needed here to guarantee a correct value.
    atomicAdd(sum, tmp_reduce);
  }
}


float average_operator(
  const float* input,
  const unsigned int size,
  cudaStream_t stream)
{
  const unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(size, threads);

  float* gpu_sum;
  cudaMalloc<float>(&gpu_sum, sizeof(float));
  cudaMemsetAsync(gpu_sum, 0, sizeof(float), stream);
  cudaStreamSynchronize(stream);

  // A SpanSize of 4 has been determined to be an optimal choice here.
  kernel_sum <4> << <blocks, threads, 0, stream >> >(
    input,
    gpu_sum,
    size);

  float cpu_sum = 0.0f;
  cudaMemcpyAsync(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(stream);

  cudaFree(gpu_sum);

  return cpu_sum /= static_cast<float>(size);
}

void unwrap(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap)
{
  const unsigned threads = 128;
  const unsigned blocks = map_blocks_to_problem(image_size, threads);

  static bool first_time = true;
  if (first_time)
  {
    kernel_extract_angle << <blocks, threads >> >(cur,
      resources->gpu_angle_predecessor_,
      image_size);
    first_time = false;
  }

  // Convert to polar notation in order to work on angles.
  kernel_extract_angle << <blocks, threads >> >(cur,
    resources->gpu_angle_current_,
    image_size);

  if (!with_unwrap)
    return;

  /* Store the new unwrapped phase image in the next buffer position.
   * The buffer is handled as a circular buffer. */
  float* next_unwrap = resources->gpu_unwrap_buffer_ + image_size * resources->next_index_;
  kernel_unwrap << < blocks, threads >> >(resources->gpu_angle_predecessor_,
    resources->gpu_angle_current_,
    next_unwrap,
    image_size);
  if (resources->size_ < resources->capacity_)
    ++resources->size_;
  resources->next_index_ = (resources->next_index_ + 1) % resources->capacity_;

  // Updating predecessor
  cudaMemcpy(resources->gpu_angle_predecessor_,
    resources->gpu_angle_current_,
    sizeof(float)* image_size,
    cudaMemcpyDeviceToDevice);

  // Applying unwrapping history to the current image.
  kernel_correct_angles << <blocks, threads >> >(
    resources->gpu_angle_current_,
    resources->gpu_unwrap_buffer_,
    image_size,
    resources->size_);
}

void unwrap_mult(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap)
{
  const unsigned threads = 128;
  const unsigned blocks = map_blocks_to_problem(image_size, threads);

  static bool first_time = true;
  if (first_time)
  {
    cudaMemcpy(resources->gpu_predecessor_,
      cur,
      sizeof(cufftComplex)* image_size,
      cudaMemcpyDeviceToDevice);
    first_time = false;
  }

  // Compute the newest phase image, not unwrapped yet
  kernel_compute_angle_mult << <blocks, threads >> >(
    resources->gpu_predecessor_,
    cur,
    resources->gpu_angle_current_,
    image_size);
  // Updating predecessor (complex image) for the next iteration
  cudaMemcpy(resources->gpu_predecessor_,
    cur,
    sizeof(cufftComplex)* image_size,
    cudaMemcpyDeviceToDevice);

  // Optional unwrapping
  if (with_unwrap)
  {
    kernel_unwrap << <blocks, threads >> >(
      resources->gpu_angle_predecessor_,
      resources->gpu_angle_current_,
      resources->gpu_unwrapped_angle_,
      image_size);
    // Updating the unwrapped angle for the next iteration.
    cudaMemcpy(
      resources->gpu_angle_predecessor_,
      resources->gpu_angle_current_,
      sizeof(float)* image_size,
      cudaMemcpyDeviceToDevice);
    // Updating gpu_angle_current_ for the rest of the function.
    cudaMemcpy(
      resources->gpu_angle_current_,
      resources->gpu_unwrapped_angle_,
      sizeof(float)* image_size,
      cudaMemcpyDeviceToDevice);
  }

  /* Copying in order to later enqueue the (not summed up with values
   * in gpu_unwrap_buffer_) phase image. */
  cudaMemcpy(
    resources->gpu_angle_copy_,
    resources->gpu_angle_current_,
    sizeof (float)* image_size,
    cudaMemcpyDeviceToDevice);

  // Applying history on the latest phase image
  kernel_correct_angles << <blocks, threads >> >(
    resources->gpu_angle_current_,
    resources->gpu_unwrap_buffer_,
    image_size,
    resources->size_);

  /* Store the new phase image in the next buffer position.
  * The buffer is handled as a circular buffer. */
  float* next_unwrap = resources->gpu_unwrap_buffer_ + image_size * resources->next_index_;
  cudaMemcpy(
    next_unwrap,
    resources->gpu_angle_copy_,
    sizeof(float)* image_size,
    cudaMemcpyDeviceToDevice);
  if (resources->size_ < resources->capacity_)
    ++resources->size_;
  resources->next_index_ = (resources->next_index_ + 1) % resources->capacity_;
}

void unwrap_diff(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size,
  const bool with_unwrap)
{
  const unsigned threads = 128;
  const unsigned blocks = map_blocks_to_problem(image_size, threads);

  static bool first_time = true;
  if (first_time)
  {
    cudaMemcpy(resources->gpu_predecessor_,
      cur,
      sizeof(cufftComplex)* image_size,
      cudaMemcpyDeviceToDevice);
    first_time = false;
  }

  // Compute the newest phase image, not unwrapped yet
  kernel_compute_angle_diff << <blocks, threads >> >(
    resources->gpu_predecessor_,
    cur,
    resources->gpu_angle_current_,
    image_size);
  //  Updating predecessor (complex image) for the next iteration
  cudaMemcpy(resources->gpu_predecessor_,
    cur,
    sizeof(cufftComplex)* image_size,
    cudaMemcpyDeviceToDevice);

  // Optional unwrapping
  if (with_unwrap)
  {
    kernel_unwrap << <blocks, threads >> >(
      resources->gpu_angle_predecessor_,
      resources->gpu_angle_current_,
      resources->gpu_unwrapped_angle_,
      image_size);
    // Updating the unwrapped angle for the next iteration.
    cudaMemcpy(
      resources->gpu_angle_predecessor_,
      resources->gpu_angle_current_,
      sizeof(float)* image_size,
      cudaMemcpyDeviceToDevice);
    // Updating gpu_angle_current_ for the rest of the function.
    cudaMemcpy(
      resources->gpu_angle_current_,
      resources->gpu_unwrapped_angle_,
      sizeof(float)* image_size,
      cudaMemcpyDeviceToDevice);
  }

  /* Copying in order to later enqueue the (not summed up with values
  * in gpu_unwrap_buffer_) phase image. */
  cudaMemcpy(
    resources->gpu_angle_copy_,
    resources->gpu_angle_current_,
    sizeof(float)* image_size,
    cudaMemcpyDeviceToDevice);

  // Applying history on the latest phase image
  kernel_correct_angles << <blocks, threads >> >(
    resources->gpu_angle_current_,
    resources->gpu_unwrap_buffer_,
    image_size,
    resources->size_);

  /* Store the new phase image in the next buffer position.
  * The buffer is handled as a circular buffer. */
  float* next_unwrap = resources->gpu_unwrap_buffer_ + image_size * resources->next_index_;
  cudaMemcpy(
    next_unwrap,
    resources->gpu_angle_copy_,
    sizeof(float)* image_size,
    cudaMemcpyDeviceToDevice);
  if (resources->size_ < resources->capacity_)
    ++resources->size_;
  resources->next_index_ = (resources->next_index_ + 1) % resources->capacity_;
}
