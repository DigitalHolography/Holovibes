#include <cmath>
#include <algorithm>

#include "tools.cuh"
#include "tools_multiply.cuh"
#include "tools.hh"
#include <device_launch_parameters.h>
#include "hardware_limits.hh"

__global__ void kernel_apply_lens(
  cufftComplex *input,
  const unsigned int input_size,
  const cufftComplex *lens,
  const unsigned int lens_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    unsigned int index2 = index % lens_size;
    input[index].x *= lens[index2].x;
    input[index].y *= lens[index2].y;
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_bursting_roi(
  const cufftComplex *input,
  const unsigned int tl_x,
  const unsigned int tl_y,
  const unsigned int br_x,
  const unsigned int br_y,
  const unsigned int curr_elt,
  const unsigned int nsamples,
  const unsigned int width,
  const unsigned int size,
  cufftComplex *output)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int width_roi = br_x - tl_x;

  // In ROI
  while (index < size)
  {
    if (index >= tl_y * width && index < br_y * width
      && index % width >= tl_x && index % width < br_x)
    {
      unsigned int x = index % width - tl_x;
      unsigned int y = index / width - tl_y;
      unsigned int index_roi = x + y * width_roi;

      output[index_roi * nsamples + curr_elt] = input[index];
    }
    index += blockDim.x * gridDim.x;
  }
}

__global__ void kernel_reconstruct_roi(
  const cufftComplex* input,
  cufftComplex* output,
  const unsigned int  input_width,
  const unsigned int  input_height,
  const unsigned int  output_width,
  const unsigned int  reconstruct_width,
  const unsigned int  reconstruct_height,
  const unsigned int  p,
  const unsigned int  nsample)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int size = reconstruct_width * reconstruct_height;

  while (index < size)
  {
    unsigned int x = index % reconstruct_width;
    unsigned int y = index / reconstruct_width;
    unsigned int x2 = x * input_width / reconstruct_width;
    unsigned int y2 = y * input_height / reconstruct_height;
    unsigned int pixel_index = y2 * input_width + x2;

    output[y * output_width + x] = input[pixel_index * nsample + p];
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Permits to shift the corners of an image.
*
* This function shift zero-frequency component to center of spectrum
* as explaines in the matlab documentation(http://fr.mathworks.com/help/matlab/ref/fftshift.html).
* The transformation happens in-place.
*/
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

/*! \brief  Kernel helper for real function
*/
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
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_log10 << <blocks, threads, 0, stream >> >(input, size);
}

/*! \brief Kernel function used in convolution_operator
*/
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
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  /* The convolution operator is used only when using autofocus feature.
   * It could be optimized but it's useless since it will be used sometimes. */
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

/*! \brief  Kernel helper for average
*/
template <unsigned SpanSize>
static __global__ void kernel_sum(const float* input, float* sum, const size_t size)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if ((index + SpanSize - 1) < size && (index % SpanSize) == 0)
  {
    float tmp_reduce = 0.0f;
    for (unsigned i = 0; i < SpanSize; ++i)
      tmp_reduce += input[index + i];
    atomicAdd(sum, tmp_reduce);
  }
}

float average_operator(
  const float* input,
  const unsigned int size,
  cudaStream_t stream)
{
  const unsigned int threads = 128;
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* gpu_sum;
  cudaMalloc<float>(&gpu_sum, sizeof(float));
  cudaMemsetAsync(gpu_sum, 0, sizeof(float), stream);
  cudaStreamSynchronize(stream);

  // SpanSize pf 4 has been determined to be an optimal choice here.
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

void copy_buffer(
  cufftComplex* src,
  cufftComplex* dst,
  const size_t nb_elts)
{
  cudaMemcpy(dst, src, sizeof(cufftComplex)* nb_elts, cudaMemcpyDeviceToDevice);
}

void unwrap(
  cufftComplex* pred,
  cufftComplex* cur,
  float* adjustments,
  const unsigned width,
  const unsigned height,
  const size_t nb_phases)
{
  const float pi = M_PI;
  const size_t size = width * height;

  // TODO : CUDA version! Here we have to work on the host.

  cufftComplex* pred_copy = new cufftComplex[size];
  cudaMemcpy(pred_copy, pred, sizeof(cufftComplex)* size, cudaMemcpyDeviceToHost);
  cufftComplex* cur_copy = new cufftComplex[size];
  cudaMemcpy(cur_copy, cur, sizeof(cufftComplex)* size, cudaMemcpyDeviceToHost);
  // Convert to polar notation in order to work on angles.
  to_polar(pred_copy, size);
  to_polar(cur_copy, size);

  float local_diff;
  float local_adjust;
  for (auto line = 0; line < height; ++line)
  {
    for (auto col = 0; col < width; ++col)
    {
      // Two-by-two diff, starting from the oldest data //
      local_diff = cur_copy[width * line + col].y - pred_copy[width * line + col].y;

      // Adjustements //
      // Equivalent phase variations in [-pi; pi)
      local_adjust = std::fmod(local_diff + pi, 2.f * pi) - pi;
      // We preserve the variation sign for pi and -pi.
      const float epsilon = 1.e-5f;
      if ((local_diff > 0.f) && (std::abs(local_adjust - pi) < epsilon))
        local_adjust = pi;

      if (std::abs(local_diff) > pi)
        local_adjust -= local_diff;
      else
        local_adjust = 0.f;

      // Cumulating the adjustement with precedent ones //
      adjustments[width * line + col] += local_adjust;
    }
  }
  // Applying the cumulated adjustements to the current frame.
  for (auto i = 0; i < size; ++i)
    cur_copy[i].y += adjustments[i];
  cudaMemcpy(cur, cur_copy, sizeof(cufftComplex)* size, cudaMemcpyHostToDevice);

  delete[] pred_copy;
  delete[] cur_copy;
}