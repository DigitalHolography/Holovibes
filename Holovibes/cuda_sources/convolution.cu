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

void convolution_kernel(float				*gpu_input,
						float				*gpu_convolved_buffer,
						cuComplex	        *cuComplex_buffer,
						CufftHandle			*plan,
						const uint			frame_width,
						const uint			frame_height,
						const cuComplex		*gpu_kernel,
						const bool			divide_convolution_enabled,
						const bool			normalize_enabled)
{
	size_t size = frame_width * frame_height;

	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	cudaXMemset(cuComplex_buffer, 0, size * sizeof(cuComplex));
	cudaSafeCall(cudaMemcpy2D(cuComplex_buffer, sizeof(cuComplex), gpu_input, sizeof(float), sizeof(float), size, cudaMemcpyDeviceToDevice));
	//At this point, cuComplex_buffer is the same as the input

	cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_FORWARD));
	//At this point, cuComplex_buffer is the FFT of the input

	kernel_multiply_frames_complex << <blocks, threads >> > (cuComplex_buffer, gpu_kernel, cuComplex_buffer, static_cast<uint>(size));
	cudaCheckError();
	//At this point, cuComplex_buffer is the FFT of the input multiplied by the FFT of the kernel

	cufftSafeCall(cufftExecC2C(plan->get(), cuComplex_buffer, cuComplex_buffer, CUFFT_INVERSE));

	kernel_complex_to_modulus << <blocks, threads >> > (cuComplex_buffer, gpu_convolved_buffer, (uint)size);
	cudaCheckError();

	if (divide_convolution_enabled)
	{
		kernel_divide_frames_float << <blocks, threads >> > (gpu_input, gpu_convolved_buffer, gpu_input, static_cast<uint>(size));
		cudaCheckError();
	}
	else
	{
		cudaXMemcpy(gpu_input, gpu_convolved_buffer, size * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}
