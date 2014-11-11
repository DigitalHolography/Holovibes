#ifndef FFT1_CUH
# define FFT1_CUH

#include <cufft.h>
#include "queue.hh"
#include "frame_desc.hh"

cufftComplex* create_lens(camera::FrameDescriptor fd, float lambda, float z);
void fft_1(int nbimages, holovibes::Queue *q, cufftComplex *lens, float *sqrt_vect, unsigned short *result_buffer, cufftHandle plan);

#endif /* !FFT1_CUH */