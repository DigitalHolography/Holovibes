#ifndef PREPROCESSING_CUH
# define PREPROCESSING_CUH

# include <cufft.h>
# include "queue.hh"

cufftComplex *make_contiguous_complex(
  holovibes::Queue& q,
  unsigned int nbimages,
  float *sqrt_vec);
float *make_sqrt_vect(int vect_size);

#endif /* !PREPROCESSING_CUH */