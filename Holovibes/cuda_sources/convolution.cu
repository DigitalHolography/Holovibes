/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once
#include "convolution.cuh"
#include "fft1.cuh"
#include "tools.cuh"
#include "tools_compute.cuh"
#include "logger.hh"
#include "Common.cuh"
#include "cuda_memory.cuh"

using holovibes::cuda_tools::CufftHandle;

void convolution_kernel(float			*gpu_input,
						float			*gpu_convolved_buffer,
						cuComplex		*cuComplex_buffer,
						CufftHandle		*plan,
						const uint		size,
						const cuComplex	*gpu_kernel,
						const bool		divide_convolution_enabled,
						const bool		normalize_enabled)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size, threads);

	/* Copy gpu_input (float*) to cuComplex_buffer (cuComplex*)
	* We only want to copy the float value as real part float number in the cuComplex_buffer
	* To skip the imaginary part, we use a pitch (skipped data) of size sizeof(float)
	*
	* The value are first all set to 0 (real & imaginary)
	* Then value are copied 1 by 1 from gpu_input into the real part
	* Imaginary is skipped and thus left to its value
	*/
	cudaXMemset(cuComplex_buffer, 0, size * sizeof(cuComplex));
	cudaSafeCall(cudaMemcpy2D(cuComplex_buffer, 	// Destination memory address
								sizeof(cuComplex), 	// Pitch of destination memory
								gpu_input, 			// Source memory address 
								sizeof(float), 		// Pitch of source memory 
								sizeof(float), 		// Width of matrix transfer (columns in bytes)
								size,				// Height of matrix transfer (rows)
								cudaMemcpyDeviceToDevice));
	// At this point, cuComplex_buffer is the same as the input

	cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_FORWARD));
	// At this point, cuComplex_buffer is the FFT of the input


	kernel_multiply_frames_complex<<<blocks, threads>>>(cuComplex_buffer, gpu_kernel, cuComplex_buffer, size);
	cudaCheckError();
	// At this point, cuComplex_buffer is the FFT of the input multiplied by the FFT of the kernel

	cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_INVERSE));

	if (divide_convolution_enabled)
	{
		kernel_complex_to_modulus<<<blocks, threads>>>(cuComplex_buffer, gpu_convolved_buffer, size);
		cudaCheckError();
		kernel_divide_frames_float<<<blocks, threads>>>(gpu_input, gpu_convolved_buffer, gpu_input, size);
	}
	else
	{
		kernel_complex_to_modulus<<<blocks, threads>>>(cuComplex_buffer, gpu_input, size);
	}
	cudaCheckError();
}