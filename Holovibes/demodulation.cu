#include "demodulation.cuh"
#include "hardware_limits.hh"
#include "tools.cuh"
#include "tools.hh"


void demodulation(
	cufftComplex* input,
	cufftComplex*                   stft_buf,
	cufftComplex*                   stft_dup_buf,
	const cufftHandle  plan1d,
	const unsigned int frame_resolution,
	const unsigned int nframes,
	cudaStream_t stream)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);

	cudaMemcpyAsync(stft_buf,
		&(stft_buf[1]),
		sizeof(cufftComplex)* (nframes * frame_resolution - 1),
		cudaMemcpyDeviceToDevice,
		stream);

	// Do the ROI
	kernel_bursting << <blocks, threads, 0, stream >> >(
		input,
		frame_resolution,
		nframes,
		stft_buf
		);

	// FFT 1D
	cufftExecC2C(plan1d, stft_buf, stft_dup_buf, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);
}