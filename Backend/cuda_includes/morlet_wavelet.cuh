#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "complex_utils.cuh"

void createMorletKernel(cuComplex* d_kernel, int N, float dt,
                        float scale, float omega0);