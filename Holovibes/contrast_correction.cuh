#ifndef CONTRAST_CORRECTION_CUH
# define CONTRAST_CORRECTION_CUH

# include <iostream>
# include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>
#include <math.h>
#include <time.h>
#include "queue.hh"
#include "hardware_limits.hh"


void correct_contrast(unsigned char *img, int img_size, int bytedepth);
void sum_histo(int *histo, int *summed_histo);


#endif /* !CONTRAST_CORRECTION_CUH */