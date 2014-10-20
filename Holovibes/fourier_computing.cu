#include "fourier_computing.cuh"

cufftComplex *do_cufft_3d(cufftComplex * input, int nbimages, int size_x, int size_y)
{
  cufftHandle plan;
  cufftPlan3d(&plan, size_x, size_y, nbimages, CUFFT_C2C);
  cufftComplex *output;
  cudaError_t er = cudaMalloc(&output, size_x * size_y * nbimages * sizeof (cufftComplex));
  cufftExecC2C(plan, input, output, CUFFT_FORWARD);
  return output;
}

cufftComplex *fft_3d(holovibes::Queue *q, int nbimages)
{
  int threads = 512;
  int blocks = (q->get_pixels() * nbimages + 511) / 512;
  cufftComplex *input = make_contigous_complex(q, nbimages);  // sqrt applied here
  dim3 lthreads(16, 16);
  dim3 lblocks(1600 / 16, 1200 / 16); // width / eight
  cufftComplex *lens;
  cudaMalloc(&lens, q->get_size() * sizeof (cufftComplex));
  kernel_quadratic_lens <<<lblocks, lthreads >>>(lens, (unsigned int) q->get_pixels(), 600.0e-9f, 2.5f);
  apply_quadratic_lens <<<blocks, threads >>>(input, q->get_pixels() * nbimages, lens, q->get_pixels());
  cudaFree(lens);
  cufftComplex *result = do_cufft_3d(input, nbimages, q->get_frame_desc().width, q->get_frame_desc().height);
  return result;
}
