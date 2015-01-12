#include "tools.cuh"

#include <device_launch_parameters.h>
#include "hardware_limits.hh"

// CONVERSION FUNCTIONS
/*! \brief  This function permit to transform an 8 bit image to her complexe representation
*
* The transformation is performed by putting the squareroot of the pixels value into the real an imagiginary
* into the complex output.
* The input image is seen as unsigned char(8 bits data container) because of her bit depth.
* The result is given in output.
*/
__global__ void img8_to_complex(
  cufftComplex* output,
  unsigned char* input,
  unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    // Image rescaling on 2^16 colors (65535 / 255 = 257)
    unsigned int val = sqrt_array[input[index] * 257];
    output[index].x = val;
    output[index].y = val;
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  This function permit to transform an 16 bit image to her complexe representation
*
* The transformation is performed by putting the squareroot of the pixels value into the real an imagiginary
* into the complex output.
* The input image is seen as unsigned short(16 bits data container) because of her bit depth.
* The result is given in output.
*/
__global__ void img16_to_complex(
  cufftComplex* output,
  unsigned short* input,
  unsigned int size,
  const float* sqrt_array)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index].x = sqrt_array[input[index]];
    output[index].y = sqrt_array[input[index]];
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Compute the modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to 
* treat with the function.
* The result is given in output.
*/
static __global__ void kernel_complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = sqrtf(input[index].x * input[index].x + input[index].y * input[index].y);

    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Compute the modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_modulus<<<blocks, threads>>>(input, output, size);
}

/*! \brief  Compute the squared modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
*/
static __global__ void kernel_complex_to_squared_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input[index].x * input[index].x + input[index].y * input[index].y;

    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Compute the squared modulus of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_squared_modulus(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_squared_modulus<<<blocks, threads>>>(input, output, size);
}

/*! \brief  Compute the arguments of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
*/
static __global__ void kernel_complex_to_argument(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float pi_div_2 = M_PI / 2.0f;
  float c = 65535.0f / M_PI;

  while (index < size)
  {
    output[index] = (atanf(input[index].y / input[index].x) + pi_div_2) * c;

    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Compute the arguments of complexe image(s) in each pixel of this.
*
* The image(s) to treat should be contigous into the input, the size is the total number of pixels to
* treat with the function.
* The result is given in output.
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void complex_to_argument(
  cufftComplex* input,
  float* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_complex_to_argument<<<blocks, threads>>>(input, output, size);
}

/*! \brief  Apply a previously computed lens to image(s).
*
* The image(s) to treat, seen as input, should be contigous, the input_size is the total number of pixels to
* treat with the function.
*/

__global__ void kernel_apply_lens(
  cufftComplex *input,
  unsigned int input_size,
  cufftComplex *lens,
  unsigned int lens_size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < input_size)
  {
    input[index].x = input[index].x * lens[index % lens_size].x;
    input[index].y = input[index].y * lens[index % lens_size].y;
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

/*! \brief  Permits to shift the corners of an image.
*
* This function shift zero-frequency component to center of spectrum
* as explaines in the matlab documentation(http://fr.mathworks.com/help/matlab/ref/fftshift.html).
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void shift_corners(
  float* input,
  unsigned int size_x,
  unsigned int size_y)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);

  kernel_shift_corners <<< lblocks, lthreads >>>(input, size_x, size_y);
}

/*! \brief  Convert the endianness of input image(s) from big endian to little endian.
*
* The image(s) to treat, seen as input, should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
*/
static __global__ void kernel_endianness_conversion(
  unsigned short* input,
  unsigned short* output,
  size_t size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = (input[index] << 8) | (input[index] >> 8);

    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Convert the endianness of input image(s) from big endian to little endian.
*
* The image(s) to treat, seen as input, should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void endianness_conversion(
  unsigned short* input,
  unsigned short* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks - 1;

  kernel_endianness_conversion <<<blocks, threads >>>(input, output, size);
}

/*! \brief  Divide all the pixels of input image(s) in complex representation by the float divider.
*
* The image(s) to treat, seen as image, should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* NB: doesn't work on architechture 2.5 in debug mod on GTX 470 graphic card
*/
__global__ void kernel_complex_divide(
  cufftComplex* image,
  unsigned int size,
  float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    image[index].x = image[index].x / divider;
    image[index].y = image[index].y / divider;
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Divide all the pixels of input image(s) by the float divider.
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* NB: doesn't work on architechture 2.5 in debug mod on GTX 470 graphic card
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
__global__ void kernel_float_divide(
  float* input,
  unsigned int size,
  float divider)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  while (index < size)
  {
    input[index] /= divider;
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  compute the log of all the pixels of input image(s).
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The value of pixels is replaced by their log10 value
*/

__global__ void kernel_log10(
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

/*! \brief  compute the log of all the pixels of input image(s).
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The value of pixels is replaced by their log10 value
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void apply_log10(
  float* input,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_log10<<<blocks, threads>>>(input, size);
}

/*! \brief  Convert all the pixels of input image(s) into unsigned short datatype.
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
*/
static __global__ void kernel_float_to_ushort(
  float* input,
  unsigned short* output,
  unsigned int size)
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

/*! \brief  Convert all the pixels of input image(s) into unsigned short datatype.
*
* The image(s) to treat should be contigous, the size is the total number of pixels to
* convert with the function.
* The result is given in output.
* This function make the Kernel call for the user in order to make the usage of the previous function easier.
*/
void float_to_ushort(
  float* input,
  unsigned short* output,
  unsigned int size)
{
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  kernel_float_to_ushort<<<blocks, threads>>>(input, output, size);
}

/*! \brief  Multiply the pixels value of 2 complexe input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_complex(
  const cufftComplex* input1,
  const cufftComplex* input2,
  cufftComplex* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index].x = input1[index].x * input2[index].x;
    output[index].y = input1[index].y * input2[index].y;
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief  Multiply the pixels value of 2 float input images
*
* The images to multiply should have the same size.
* The result is given in output.
* Output should have the same size of inputs.
*/
__global__ void kernel_multiply_frames_float(
  const float* input1,
  const float* input2,
  float* output,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  while (index < size)
  {
    output[index] = input1[index] * input2[index];
    index += blockDim.x * gridDim.x;
  }
}


/*! \brief  apply the convolution operator to 2 complex images (x,k).
*
* The 2 images should have the same size.
* The result value is given is out.
* The 2 used planes should be externally prepared (for performance reasons).
* For further informations: Autofocus of holograms based on image sharpness.
*/
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

  kernel_multiply_frames_complex <<<blocks, threads>>>(tmp_x, tmp_k, tmp_x, size);

  cudaDeviceSynchronize();

  cufftExecC2C(plan2d_x, tmp_x, tmp_x, CUFFT_INVERSE);

  cudaDeviceSynchronize();

  kernel_complex_to_modulus <<<blocks, threads>>>(tmp_x, out, size);

  cudaFree(tmp_x);
  cudaFree(tmp_k);
}

/*! \brief  Extract a part of the input image to the output.
*
* The exracted aera should be less Than the input image.
* The result extracted image given is contained in output, the output should be preallocated.
* Coordonates of the extracted zone are specified into the zone.
*/
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

/*! \brief  Sum all the pixels of the input image.
*
* The result of the sumation is contained in the parameted sum,
* The size parameter represent the number of pixels to sum,
* it should be equal to the number of pixels of the image.
*/
static __global__ void kernel_sum(
  const float* input,
  float* sum,
  unsigned int size)
{
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (index < size)
  {
    atomicAdd(sum, input[index]);
    index += blockDim.x * gridDim.x;
  }
}

/*! \brief   Make the average of all pixels contained into the input image
*
* The size parameter is the number of pixels of the input image
*/
#include <iostream>
float average_operator(
  const float* input,
  const unsigned int size)
{
  const unsigned int threads = get_max_threads_1d();
  const unsigned int max_blocks = get_max_blocks();
  unsigned int blocks = (size + threads - 1) / threads;

  if (blocks > max_blocks)
    blocks = max_blocks;

  float* gpu_sum;
  cudaMalloc<float>(&gpu_sum, sizeof(float));
  cudaMemset(gpu_sum, 0, sizeof(float));

  kernel_sum <<<blocks, threads>>>(
    input,
    gpu_sum,
    size);

  float cpu_sum = 0.0f;
  cudaMemcpy(&cpu_sum, gpu_sum, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(gpu_sum);

  cpu_sum /= float(size);

  return cpu_sum;
}
