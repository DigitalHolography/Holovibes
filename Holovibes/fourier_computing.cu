#include "fourier_computing.cuh"

cufftComplex *do_cufft_3d(cufftComplex *input, int nbimages, int size_x, int size_y)
{
  cufftHandle plan;
  cufftPlan3d(&plan, size_x, size_y, nbimages, CUFFT_C2C);
  cufftComplex *output;
  cudaMalloc(&output, size_x * size_y * nbimages * sizeof (cufftComplex));
  cufftExecC2C(plan, input, output, CUFFT_FORWARD);
  return output;
}

cufftComplex *fft_3d(holovibes::Queue *q, int nbimages)
{
  int threads = 512;
  int blocks = (q->get_pixels() * nbimages + threads - 1) / threads;

  if (blocks > 65536)
    blocks = 65536;

  cufftComplex *input = make_contigous_complex(q, nbimages);  // sqrt applied here

  // Constructing lens
  dim3 lthreads(16, 16);
  dim3 lblocks(q->get_frame_desc().width / 16, q->get_frame_desc().height / 16); // width / height
  cufftComplex *lens;
  cudaMalloc(&lens, q->get_pixels() * sizeof (cufftComplex));
  kernel_quadratic_lens << <lblocks, lthreads >> >(lens, q->get_frame_desc().width, q->get_frame_desc().height, 532.0e-9f, 1.36f);

  // Applying lens
  apply_quadratic_lens <<<blocks, threads >>>(input, q->get_pixels() * nbimages, lens, q->get_pixels());
  cudaFree(lens);

  // Applying FFT
  cufftComplex *result = do_cufft_3d(input, nbimages, q->get_frame_desc().width, q->get_frame_desc().height);
  return result;
}
