#include "fft1.cuh"
#include "hardware_limits.hh"
#include "frame_desc.hh"
#include "tools.hh"
#include "tools.cuh"
#include "preprocessing.cuh"
#include "transforms.cuh"

void fft1_lens(
  cufftComplex* lens,
  const camera::FrameDescriptor& fd,
  const float lambda,
  const float z,
  cudaStream_t stream)
{
  unsigned int threads = 128;
  unsigned int blocks = map_blocks_to_problem(fd.frame_res(), threads);

  kernel_quadratic_lens << <blocks, threads, 0, stream >> >(lens, fd, lambda, z);
}

void fft_1(
	cufftComplex* input,
	const cufftComplex* lens,
	const cufftHandle plan1D,
	const cufftHandle plan2D,
	const unsigned int frame_resolution,
	const unsigned int nframes,
	const unsigned int p,
	const unsigned int q,
	cudaStream_t stream)
{
	const unsigned int n_frame_resolution = frame_resolution * nframes;

	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);
	
	cufftComplex* pframe = input + frame_resolution * p;
	cufftComplex* qframe = input + frame_resolution * q;

	cufftExecC2C(plan1D, input, input, CUFFT_FORWARD);

	// Apply lens on multiple frames.
	kernel_apply_lens << <blocks, threads, 0, stream >> >(pframe, frame_resolution, lens, frame_resolution);
	cudaStreamSynchronize(stream);
	// FFT

    cufftExecC2C(plan2D, pframe, pframe, CUFFT_FORWARD);

	if (p != q)
	{
		kernel_apply_lens << <blocks, threads, 0, stream >> >(qframe, frame_resolution, lens, frame_resolution);
		cufftExecC2C(plan2D, qframe, qframe, CUFFT_FORWARD);
	}

	cudaStreamSynchronize(stream);
}
/*
void fft_1(
  cufftComplex* input,
  const cufftComplex* lens,
  const cufftHandle plan,
  const unsigned int frame_resolution,
  const unsigned int nframes,
  cudaStream_t stream)
{
  const unsigned int n_frame_resolution = frame_resolution * nframes;

  unsigned int threads = get_max_threads_1d();
  unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);

  // Apply lens on multiple frames.
  kernel_apply_lens << <blocks, threads, 0, stream >> >(input, n_frame_resolution, lens, frame_resolution);

  cudaStreamSynchronize(stream);
  // FFT
  cufftResult res = cufftExecC2C(plan, input, input, CUFFT_FORWARD);
  
  cudaStreamSynchronize(stream);
}*/
