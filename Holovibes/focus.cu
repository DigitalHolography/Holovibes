#include "focus.cuh"

static __global__ void kernel_complex_2_to_pow(cufftComplex* input,unsigned int size, int power)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    input[index].x = pow((double)input[index].x, power);
    input[index].y = input[index].x;
    index += blockDim.x * gridDim.x;
  }
}

void complex_to_pow(cufftComplex *input, unsigned int size, int power)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;
  kernel_complex_2_to_pow<<<blocks,threads>>>(input, size, power);
}

static __global__  void kernel_average_complex(cufftComplex* input, float *output, unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    atomicAdd(output, input[index].x);
  }
}

float average_complex_2d(cufftComplex* input, unsigned int size) // <>
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;
  float sum_value;
  kernel_average_complex<<<blocks,threads>>>(input, &sum_value, size);
  return (sum_value / (float)size);
}


/*cufftComplex *output; // should be done once
cudaMalloc(&output, 2048 * 2048 * sizeof(cufftComplex)); //should be done once
cudaMemset(output, 0, 2048 * 2048 * sizeof(cufftComplex)); // should be done everytime the s zone*/

void extract_s(cufftComplex* input, cufftComplex *output,
  unsigned int size, holovibes::Rectangle& s_coord,
  camera::FrameDescriptor input_fd)
{
  int start_x = s_coord.top_left.x;
  int start_y = s_coord.top_left.y;
  int end_x = s_coord.top_right.x;
  int end_y = s_coord.bottom_left.y;
  int s_size_x = end_x - start_x;
  int y_output = 0;
  for (int y = start_y; y < end_y; y++)
  {
    cudaMemcpy(output + y_output * input_fd.width,
      input + y * input_fd.width + start_x,
      s_size_x * sizeof(cufftComplex),
      cudaMemcpyDeviceToDevice);
    y_output++;
  }
}

float compute_v(cufftComplex *s, unsigned int size) //extracted s
{
  cufftComplex *square_s;
  cudaMalloc(&square_s, size * sizeof(cufftComplex));
  cudaMemcpy(square_s, s, size * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
  complex_to_pow(square_s, size, 2);
  float s_av = average_complex_2d(s, size);// <I>
  float s_square_av = average_complex_2d(square_s, size);//<I^2>
  cudaFree(square_s);
  return (s_square_av - (s_av * s_av));
}

static __global__ void kernel_multiply_matrixes(cufftComplex *input1, cufftComplex *input2,
  cufftComplex *output, unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    output[index].x = input1[index].x * input2[index].x;
    output[index].y = input1[index].y * input2[index].y;
  }
}
// plan should be allocated at size of
void convolution(cufftComplex *x, cufftComplex *k, cufftHandle plan2d, unsigned int size)
{
  cufftComplex *to_invert;
  cudaMalloc(&to_invert, size);
  cufftExecC2C(plan2d, x, x, CUFFT_FORWARD);
  cufftExecC2C(plan2d, k, k, CUFFT_FORWARD);
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;
  kernel_multiply_matrixes<<<blocks,threads>>>(x, k, to_invert, size);
  cufftExecC2C(plan2d, to_invert, to_invert, CUFFT_INVERSE);
  cudaFree(to_invert);
}



