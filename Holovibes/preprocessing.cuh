#ifndef PREPROCESSING_CUH
# define PREPROCESSING_CUH

# include <cufft.h>
# include "queue.hh"

void make_sqrt_vect(float* out, unsigned short n);

void make_contiguous_complex(
  holovibes::Queue& input,
  cufftComplex* output,
  unsigned int n,
  const float* sqrt_array);

#endif /* !PREPROCESSING_CUH */