#ifndef VIBROMETRY_CUH
# define VIBROMETRY_CUH

# include <cufft.h>

void frame_ratio(
  cufftComplex* frame_p,
  cufftComplex* frame_q,
  cufftComplex* output,
  unsigned int size);

#endif /* !VIBROMETRY_CUH */