#ifndef FOURIER_COMPUTING_CUH
# define FOURIER_COMPUTING_CUH

# include "tools.cuh"
# include "queue.hh"
# include "preprocessing.cuh"
# include "transforms.cuh"
# include <cufft.h>
# include <cufftXt.h>
# include <cufftw.h>

cufftComplex *fft_3d(holovibes::Queue *q, int nbimages);

#endif /* !FOURIER_COMPUTING_CUH */