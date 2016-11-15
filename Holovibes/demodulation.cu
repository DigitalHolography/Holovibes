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
	const unsigned int pindex,
	cudaStream_t stream)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(frame_resolution, threads);

	// Do the ROI
	/*kernel_bursting << <blocks, threads, 0, stream >> >(
		input,
		frame_resolution,
		nframes,
		stft_buf
		);*/

	// FFT 1D
	cufftExecC2C(plan1d, input, input, CUFFT_FORWARD);
	//cudaStreamSynchronize(stream);

	/*kernel_reconstruct << <blocks, threads, 0, stream >> >(
		stft_dup_buf,
		input,
		pindex,
		nframes,
		frame_resolution
		);
		*/
	//cudaStreamSynchronize(stream);

}