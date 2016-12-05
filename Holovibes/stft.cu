#include <device_launch_parameters.h>

#include "stft.cuh"
#include "hardware_limits.hh"
#include "geometry.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"
#include "geometry.hh"

void stft(
  cufftComplex*                   input,
  cufftComplex*                   gpu_queue,
  cufftComplex*                   stft_buf,
  const cufftHandle               plan1d,
  unsigned int                    stft_level,
  unsigned int                    p, 
  unsigned int                    q, 
  unsigned int                    frame_size,
  cudaStream_t                    stream)
{
  unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(frame_size, threads);


  // FFT 1D
  cufftExecC2C(plan1d, gpu_queue, stft_buf, CUFFT_FORWARD);
  cudaStreamSynchronize(stream);

  cudaMemcpy(
	  input,
	  stft_buf + p * frame_size,
	  sizeof(cufftComplex)* frame_size,
	  cudaMemcpyDeviceToDevice);

  if (p != q)
  {
	  cudaMemcpy(
		  input + frame_size,
		  stft_buf + q * frame_size,
		  sizeof(cufftComplex)* frame_size,
		  cudaMemcpyDeviceToDevice);

  }

}