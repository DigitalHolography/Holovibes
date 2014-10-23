#ifndef PREPROCESSING_CUH
#define PREPROCESSING_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftw.h>
#include "queue.hh"
#include "tools.cuh"

cufftComplex *make_contigous_complex(holovibes::Queue *q, int nbimages);

#endif