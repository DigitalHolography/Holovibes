#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cmath>


#include "convolution.cuh"
#include "tools_conversion.cuh"
#include "tools_analysis.cuh"
#include "unique_ptr.hh"
#include "tools_compute.cuh"
#include "cuda_memory.cuh"
#include "logger.hh"
#include "cuComplex.h"
#include "cufft_handle.hh"
#include "vascular_pulse.cuh"

void compute_first_correlation()
{

}