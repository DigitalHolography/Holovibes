#include "fft1.cuh"

cufftComplex* create_lens(camera::FrameDescriptor fd, float lambda, float z)
{
  unsigned int threads_2d = get_max_threads_2d();
  dim3 lthreads(threads_2d, threads_2d);
  dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);
  cufftComplex *lens;
  cudaMalloc(&lens, fd.width * fd.height * sizeof(cufftComplex));
  kernel_quadratic_lens <<<lblocks, lthreads>>>(lens, fd, lambda, z);

  return lens;
}

void fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan)
{
  // Sizes
  unsigned int pixel_size = q->get_frame_desc().width * q->get_frame_desc().height * nbimages;
  unsigned int complex_size = pixel_size * sizeof(cufftComplex);
  unsigned int short_size = pixel_size * sizeof(unsigned short);

  // Loaded images --> complex
  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = (pixel_size + threads - 1) / threads;

  // Hardware limit !!
  if (blocks > get_max_blocks())
    blocks = get_max_blocks();

  cufftComplex* complex_input = make_contigous_complex(q, nbimages, sqrt_vect);

  // Apply lens
  apply_quadratic_lens <<<blocks, threads>>>(complex_input, pixel_size, lens, q->get_pixels());

  // FFT
  cufftExecC2C(plan, complex_input, complex_input, CUFFT_FORWARD);

  // Complex --> real (unsigned short)
  complex_2_module <<<blocks, threads>>>(complex_input, result_buffer, pixel_size);

  // Free all
  cudaFree(complex_input);
}