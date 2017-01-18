#include <cmath>
#include <algorithm>
#include <device_launch_parameters.h>

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_conversion.cuh"
#include "tools_unwrap.cuh"
#include "tools.hh"
#include "geometry.hh"
#include "hardware_limits.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"

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
  unsigned int size_x2 = size_x >> 1;
  unsigned int size_y2 = size_y >> 1;
  if (j >= size_y2)
  {
    // Left superior quarter of the matrix
    if (i < size_x2)
    {
      ni = i + size_x2;
      nj = j - size_y2;
    }
    // Right superior quarter
    else
    {
      ni = i - size_x2;
      nj = j - size_y2;
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

void phase_increase(
  const cufftComplex* cur,
  holovibes::UnwrappingResources* resources,
  const size_t image_size)
{
  const unsigned threads = 128; // 3072 cuda cores / 24 SMM = 128 Threads per SMM
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

void unwrap_2d(
	float *input,
	const cufftHandle plan2d,
	holovibes::UnwrappingResources_2d *res,
	camera::FrameDescriptor& fd,
	float *output,
	cudaStream_t stream)
{
	unsigned int threads_2d = get_max_threads_2d();
	dim3 lthreads(threads_2d, threads_2d);
	dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);
	const unsigned threads = 128;
	const unsigned blocks = map_blocks_to_problem(res->image_resolution_, threads);

	kernel_init_unwrap_2d << < lblocks, lthreads, 0, stream >> > (
		fd.width,
		fd.height,
		fd.frame_res(),
		input,
		res->gpu_fx_,
		res->gpu_fy_,
		res->gpu_z_);
	unsigned short middlex = fd.width >> 1;
	unsigned short middley = fd.height >> 1;
	circ_shift_float << < blocks, threads, 0, stream >> > (
		res->gpu_fx_,
		res->gpu_shift_fx_,
		middlex,
		middley,
		fd.width,
		fd.height,
		fd.frame_res());
	circ_shift_float << < blocks, threads, 0, stream >> > (
		res->gpu_fy_,
		res->gpu_shift_fy_,
		middlex,
		middley,
		fd.width,
		fd.height,
		fd.frame_res());
	gradient_unwrap_2d(plan2d, res, fd, stream);
	eq_unwrap_2d(plan2d, res, fd, stream);
	phi_unwrap_2d(plan2d, res, fd, output, stream);
}

void gradient_unwrap_2d(
	const cufftHandle plan2d,
	holovibes::UnwrappingResources_2d *res,
	camera::FrameDescriptor& fd,
	cudaStream_t stream)
{
	const unsigned threads = 128;
	const unsigned blocks = map_blocks_to_problem(res->image_resolution_, threads);
	cufftComplex single_complex = make_cuComplex(0.f, static_cast<float>(M_2PI));

	cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
	cufftExecC2C(plan2d, res->gpu_z_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
	kernel_multiply_complexes_by_floats_ <<< blocks, threads, 0, stream>>>(
		res->gpu_shift_fx_,
		res->gpu_shift_fy_,
		res->gpu_grad_eq_x_,
		res->gpu_grad_eq_y_,
		fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_INVERSE);
	kernel_multiply_complexes_by_single_complex << < blocks, threads, 0, stream >> >(
		res->gpu_grad_eq_x_,
		res->gpu_grad_eq_y_,
		single_complex,
		fd.frame_res());
}

void eq_unwrap_2d(
	const cufftHandle plan2d,
	holovibes::UnwrappingResources_2d *res,
	camera::FrameDescriptor& fd,
	cudaStream_t stream)
{
	const unsigned threads = 128;
	const unsigned blocks = map_blocks_to_problem(res->image_resolution_, threads);
	cufftComplex single_complex = make_cuComplex(0, 1);

	kernel_multiply_complex_by_single_complex << < blocks, threads, 0, stream >> >(
		res->gpu_z_,
		single_complex,
		fd.frame_res());
	kernel_conjugate_complex << < blocks, threads, 0, stream >> >(
		res->gpu_z_,
		fd.frame_res());
	kernel_multiply_complex_frames_by_complex_frame << < blocks, threads, 0, stream >> >(
		res->gpu_grad_eq_x_,
		res->gpu_grad_eq_y_,
		res->gpu_z_,
		fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_FORWARD);
	cufftExecC2C(plan2d, res->gpu_grad_eq_y_, res->gpu_grad_eq_y_, CUFFT_FORWARD);
	kernel_norm_ratio << < blocks, threads, 0, stream >> >(
		res->gpu_shift_fx_,
		res->gpu_shift_fy_,
		res->gpu_grad_eq_x_,
		res->gpu_grad_eq_y_,
		fd.frame_res());
}

void phi_unwrap_2d(
	const cufftHandle plan2d,
	holovibes::UnwrappingResources_2d *res,
	camera::FrameDescriptor& fd,
	float *output,
	cudaStream_t stream)
{
	const unsigned threads = 128;
	const unsigned blocks = map_blocks_to_problem(res->image_resolution_, threads);

	//	kernel_convergence << < 1, 1, 0, stream >> >(res->gpu_grad_eq_x_,
	//		res->gpu_grad_eq_y_);
	kernel_add_complex_frames << < blocks, threads, 0, stream >> >(
		res->gpu_grad_eq_x_,
		res->gpu_grad_eq_y_,
		fd.frame_res());
	cufftExecC2C(plan2d, res->gpu_grad_eq_x_, res->gpu_grad_eq_x_, CUFFT_INVERSE);

	kernel_unwrap2d_last_step << < blocks, threads, 0, stream >> > (
	output,
	res->gpu_grad_eq_x_,
	fd.frame_res());
}

__global__ void circ_shift(
	cufftComplex *input,
	cufftComplex *output,
	const int i, // shift on x axis
	const int j, // shift on y axis
	const unsigned int width,
	const unsigned int height,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index_x = 0;
	int index_y = 0;
	int shift_x = 0;
	int shift_y = 0;
	// In ROI
	while (index < size)
	{
		index_x = index % width;
		index_y = index / height;
		shift_x = index_x - i;
		shift_y = index_y - j;
		if (shift_x < 0)
			shift_x = width + shift_x;
		if (shift_y < 0)
			shift_y = height + shift_y;
		output[(width * shift_y) + shift_x] = input[index];
		index += blockDim.x * gridDim.x;
	}
}

__global__ void circ_shift_float(
	float *input,
	float *output,
	const int i, // shift on x axis
	const int j, // shift on y axis
	const unsigned int width,
	const unsigned int height,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index_x = 0;
	int index_y = 0;
	int shift_x = 0;
	int shift_y = 0;
	// In ROI
	while (index < size)
	{
		index_x = index % width;
		index_y = index / height;
		shift_x = index_x - i;
		shift_y = index_y - j;
		if (shift_x < 0)
			shift_x = width + shift_x;
		if (shift_y < 0)
			shift_y = height + shift_y;
		output[(width * shift_y) + shift_x] = input[index];
		index += blockDim.x * gridDim.x;
	}
}