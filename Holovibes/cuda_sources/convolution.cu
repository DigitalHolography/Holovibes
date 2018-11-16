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
	uint threads = 1024;
	uint blocks = map_blocks_to_problem(size, threads);
	
	holovibes::cuda_tools::UniquePtr<float> output(threads);
	normalize_float_matrix<1024><<<blocks, threads, blocks * sizeof(float)>>>(gpu_kernel_buffer_, output, size);
	
	float output_cpu[4096]; //never more than 4096 threads
	//need to be on cpu for the sum
	cudaMemcpy(output, output_cpu, threads * sizeof(float), cudaMemcpyDeviceToHost);
	float sum = 0;
	for (int i = 0; i < threads; i++)
		sum += output_cpu[i];

	gpu_float_divide(gpu_kernel_buffer_, size, sum);
}

void convolution_kernel(const float		*input,
						float			*output,
						const uint		frame_width,
						const uint		frame_height,
						const float		*kernel)
{
	
	size_t size = frame_width * frame_height;

	uint	threads = get_max_threads_1d();
	uint	blocks = map_blocks_to_problem(size, threads);

	holovibes::cuda_tools::UniquePtr<cuComplex> output_fft(size);
	holovibes::cuda_tools::UniquePtr<cuComplex> output_kernel(size);
	if (!output_fft || !output_kernel)
	{
		std::cout << "Couldn't allocate buffers for convolution.\n";
		return;
	}

	holovibes::cuda_tools::UniquePtr<cuComplex> tmp_complex(size);
	cudaMemset(tmp_complex.get(), 0, size * sizeof(cuComplex));
	cudaCheckError();
	cudaMemcpy2D(tmp_complex.get(), sizeof(cuComplex), input, sizeof(float), sizeof(float), size, cudaMemcpyDeviceToDevice);
	CufftHandle plan(frame_width, frame_height, CUFFT_C2C);
	cufftExecC2C(plan, tmp_complex.get(), output_fft.get(), CUFFT_FORWARD);

	cudaMemcpy2D(tmp_complex.get(), sizeof(cuComplex), kernel, sizeof(float), sizeof(float), size, cudaMemcpyDeviceToDevice);
	cufftExecC2C(plan, tmp_complex.get(), output_kernel.get(), CUFFT_FORWARD);

	kernel_multiply_frames_complex <<<blocks, threads>>>(output_fft, output_kernel, output_fft, size);

	cufftExecC2C(plan, output_fft, output_fft, CUFFT_INVERSE);

	kernel_complex_divide<<<blocks, threads>>>(output_fft, size, 1e13);
	
	kernel_complex_to_modulus <<<blocks, threads>>>(output_fft, output, (uint) size);

	kernel_divide_frames_float << <blocks, threads>> > (input, output, output, size);
	//print_float << <blocks, threads >> > (output);
}
