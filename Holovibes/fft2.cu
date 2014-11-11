#include "fft2.cuh"

cufftComplex *create_spectral(float lambda, float distance, int size_x, int size_y, float pasx, float pasy, camera::FrameDescriptor fd)
{
  cufftComplex *output;
  cudaMalloc(&output, size_x * size_y * sizeof(cufftComplex));
  cudaMemset(output, 0, size_x * size_y * sizeof(cufftComplex));

  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(size_x / threads_2d, size_y / threads_2d);
  kernel_spectral_lens<<<lblocks, lthreads>>>(output, fd, lambda, distance);

  return output;
}

void fft_2(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan3d, unsigned int p, cufftHandle plan2d)
{
  // Sizes
  unsigned int pixel_size = q->get_pixels() * nbimages;
  unsigned int complex_size = pixel_size * sizeof(cufftComplex);
  unsigned int image_pixel = q->get_pixels();
  unsigned int complex_image_size = image_pixel * sizeof(cufftComplex);
  unsigned int short_size = pixel_size * sizeof(unsigned short);
  unsigned short size_x = q->get_frame_desc().width;
  unsigned short size_y = q->get_frame_desc().height;

  // Loaded images --> complex
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (image_pixel + threads - 1) / threads; //one image blocks

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks() - 1;

  //get contigous images
  cufftComplex* complex_input = make_contigous_complex(q, nbimages, sqrt_vect);

  //3d fft
  cufftExecC2C(plan3d, complex_input, complex_input, CUFFT_FORWARD);

  // extratct the (p) image
  cufftComplex* pimage;
  cudaMalloc(&pimage, complex_image_size);
  cufftComplex *image = complex_input + p * image_pixel;
  cudaMemcpy(pimage, image, complex_image_size, cudaMemcpyDeviceToDevice);

  // apply lens
  apply_quadratic_lens << <blocks, threads >> >(pimage, image_pixel, lens, image_pixel);

  if (cufftExecC2C(plan2d, pimage, pimage, CUFFT_INVERSE) != CUFFT_SUCCESS)
    std::cout << "fail fft 2" << std::endl;
  cudaDeviceSynchronize();

  divide<<<blocks, threads >>>(pimage, size_x, size_y, nbimages);

  //back to real
  complex_2_module <<<blocks, threads >> >(pimage, result_buffer, image_pixel); // one image

  cudaFree(pimage);
  cudaFree(complex_input);
}