#pragma once
#include "complex_utils.cuh"
#include "morlet_wavelet.cuh"
#include "frame_desc.hh"

void wavelet_transform(cuComplex* output, cuComplex* input, const cufftHandle plan1d,  const camera::FrameDescriptor& fd, int tranformation_size, const cudaStream_t stream);