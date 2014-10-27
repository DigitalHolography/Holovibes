#include "fourier_computing.cuh"

cufftComplex *do_cufft_3d(cufftComplex *input, int nbimages, int size_x, int size_y)
{
  cufftHandle plan;
  cufftPlan3d(&plan, size_x, size_y, nbimages, CUFFT_C2C);
  cufftExecC2C(plan, input, input, CUFFT_FORWARD);
  return input;
}
