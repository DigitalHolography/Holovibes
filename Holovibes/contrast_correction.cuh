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


void manual_contrast_correction(void *img, unsigned int img_size, int bytedepth, unsigned int manual_min, unsigned int manual_max);
void auto_contrast_correction(unsigned int *min, unsigned int *max, void *img, unsigned int img_size, unsigned int bytedepth, unsigned int percent);
void sum_histo(int *histo, int *summed_histo);


#endif /* !CONTRAST_CORRECTION_CUH */