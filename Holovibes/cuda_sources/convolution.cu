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
using holovibes::cuda_tools::CufftHandle;

//the three next function are for test
__global__
void print_kernel(cuComplex *output)
{
	if (threadIdx.x < 32)
		printf("%d, %f, %f\n", threadIdx.x, output[threadIdx.x].x, output[threadIdx.x].y);
}

__global__
void print_float(float *output)
{
	if (threadIdx.x < 32)
		printf("%d, %f\n", threadIdx.x, output[threadIdx.x]);
}

__global__
void fill_output(float *out, unsigned size)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		out[idx] = 65000.f;
	}
}

void normalize_kernel(float		*gpu_kernel_buffer_,
					  size_t	size)
{
	float sum = get_norm(gpu_kernel_buffer_, size);

	// Calling an empty function, why?
	// gpu_float_divide(gpu_kernel_buffer_, static_cast<uint>(size), sum);
}

void convolution_kernel(float		*gpu_input,
	float			*gpu_convolved_buffer,
	CufftHandle		*plan,
	const uint		frame_width,
	const uint		frame_height,
	const float		*gpu_kernel,
	const bool		divide_convolution_enabled,
	const bool		normalize_enabled)
{
	size_t size = frame_width * frame_height;

	float norm_input;
	if (normalize_enabled)
		norm_input = get_norm(gpu_input, size);

	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	holovibes::cuda_tools::UniquePtr<cuComplex> output_fft(size);
	holovibes::cuda_tools::UniquePtr<cuComplex> output_kernel(size);
	if (!output_fft || !output_kernel)
	{
		LOG_ERROR("Couldn't allocate buffers for convolution.\n");
		return;
	}

	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_complex(size);
	cudaMemset(tmp_complex.get(), 0, size * sizeof(cuComplex));
	cudaCheckError();
	cudaMemcpy2D(tmp_complex.get(), sizeof(cuComplex), gpu_input, sizeof(float), sizeof(float), size, cudaMemcpyDeviceToDevice);
	cufftExecC2C(plan->get(), tmp_complex.get(), output_fft.get(), CUFFT_FORWARD);

	cudaMemcpy2D(tmp_complex.get(), sizeof(cuComplex), gpu_kernel, sizeof(float), sizeof(float), size, cudaMemcpyDeviceToDevice);
	cufftExecC2C(plan->get(), tmp_complex.get(), output_kernel.get(), CUFFT_FORWARD);

	kernel_multiply_frames_complex << <blocks, threads >> > (output_fft, output_kernel, output_fft, static_cast<uint>(size));

	cufftExecC2C(plan->get(), output_fft, output_fft, CUFFT_INVERSE);

	kernel_complex_to_modulus << <blocks, threads >> > (output_fft, gpu_convolved_buffer, (uint)size);

	if (divide_convolution_enabled)
		kernel_divide_frames_float << <blocks, threads >> > (gpu_input, gpu_convolved_buffer, gpu_input, static_cast<uint>(size));
	else
		cudaMemcpy(gpu_input, gpu_convolved_buffer, size * sizeof(float), cudaMemcpyDeviceToDevice);

	if (normalize_enabled) {
		float norm_output = get_norm(gpu_input, size);
		gpu_multiply_const(gpu_input, static_cast<uint>(size), (norm_input / norm_output));
	}

}
