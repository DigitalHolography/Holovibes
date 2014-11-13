#ifndef PREPROCESSING_CUH
# define PREPROCESSING_CUH

# include <cufft.h>
# include "queue.hh"

cufftComplex *make_contiguous_complex(
  holovibes::Queue& q,
  unsigned int nbimages,
  float *sqrt_vec);
void make_sqrt_vect(float* out, unsigned short n);

#endif /* !PREPROCESSING_CUH */