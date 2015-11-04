#include "tools.cuh"
#include "tools_multiply.cuh"

#include <device_launch_parameters.h>
#include "hardware_limits.hh"

__global__ void kernel_apply_lens(
  cufftComplex *input,
  unsigned int input_size,
  cufftComplex *lens,
  unsigned int lens_size)
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
  cufftComplex *input,
  unsigned int tl_x,
  unsigned int tl_y,
  unsigned int br_x,
  unsigned int br_y,
  unsigned int curr_elt,
  unsigned int nsamples,
  unsigned int width,
  unsigned int size,
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
  cufftComplex* input,
  cufftComplex* output,
  unsigned int  input_width,
  unsigned int  input_height,
  unsigned int  output_width,
  unsigned int  reconstruct_width,
  unsigned int  reconstruct_height,
  unsigned int  p,
  unsigned int  nsample)
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
  unsigned int size_x,
  unsigned int size_y)
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
  unsigned int size_x,
  unsigned int size_y)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

  kernel_shift_corners << < lblocks, lthreads >> >(input, size_x, size_y);
}

/*! \brief  Kernel helper for real function
*/
static __global__ void kernel_log10(
  float* input,
  unsigned int size)
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
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_log10 << <blocks, threads >> >(input, size);
}

/*! \brief Kernel function used in convolution_operator
*/
static __global__ void kernel_complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
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
  unsigned int size,
  cufftHandle plan2d_x,
  cufftHandle plan2d_k)
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

  cudaDeviceSynchronize();

  kernel_multiply_frames_complex << <blocks, threads >> >(tmp_x, tmp_k, tmp_x, size);

  cudaDeviceSynchronize();

  cufftExecC2C(plan2d_x, tmp_x, tmp_x, CUFFT_INVERSE);

  cudaDeviceSynchronize();

  kernel_complex_to_modulus << <blocks, threads >> >(tmp_x, out, size);

  cudaFree(tmp_x);
  cudaFree(tmp_k);
}

void frame_memcpy(
  const float* input,
  const holovibes::Rectangle& zone,
  const unsigned int input_width,
  float* output,
  const unsigned int output_width)
{
  const unsigned int zone_width = abs(zone.top_right.x - zone.top_left.x);
  const unsigned int zone_height = abs(zone.bottom_left.y - zone.top_left.y);

  const float* zone_ptr = input + (zone.top_left.y * input_width + zone.top_left.x);

  cudaMemcpy2D(
    output,
    output_width * sizeof(float),
    zone_ptr,
    input_width * sizeof(float),
    zone_width * sizeof(float),
    zone_height,
    cudaMemcpyDeviceToDevice);
}

/*! \brief  Kernel helper for average
*/
template <unsigned SpanSize>
static __global__ void kernel_sum(const float* input, float* sum, size_t size)
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
  const unsigned int size)
{
  const unsigned int threads = 128;
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* gpu_sum;
  cudaMalloc<float>(&gpu_sum, sizeof(float));
  cudaMemset(gpu_sum, 0, sizeof(float));

  // SpanSize pf 4 has been determined to be an optimal choice here.
  kernel_sum <4> << <blocks, threads >> >(
    input,
    gpu_sum,
    size);

  float cpu_sum = 0.0f;
  cudaMemcpy(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpu_sum);
  cpu_sum /= float(size);
  return cpu_sum;
}

void copy_buffer(
  cufftComplex* src,
  cufftComplex* dst,
  const size_t nb_elts)
{
  cudaMemcpy(dst, src, sizeof(cufftComplex)* nb_elts, cudaMemcpyDeviceToDevice);
}