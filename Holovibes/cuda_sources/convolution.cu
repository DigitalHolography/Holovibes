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

#include "convolution.cuh"

__global__
void kernel_multiply_kernel(cuComplex	*input,
							cuComplex	*gpu_special_queue,
							const uint	gpu_special_queue_buffer_length,
							const uint	frame_resolution,
							const uint	i_width,
							const float	*kernel,
							const uint	k_width,
							const uint	k_height,
							const uint	nsamples,
							const uint	start_index,
							const uint	max_index)
{
	uint index = blockIdx.x * blockDim.x + threadIdx.x;
	//uint size = frame_resolution * nsamples;
	uint k_size = k_width * k_height;
	while (index < frame_resolution)
	{
		cuComplex sum = make_cuComplex(0, 0);

		for (uint z = 0; z < nsamples; ++z)
		for (uint m = 0; m < k_width; ++m)
		for (uint n = 0; n < k_height; ++n) {
			cuComplex a = gpu_special_queue[(index + m + n * i_width + (((z + start_index) % max_index) * frame_resolution)) % gpu_special_queue_buffer_length];
			float b = kernel[m + n * k_width + (z * k_size)];
			sum.x += a.x * b;
			sum.y += a.y * b;
		}
		const uint n_k_size = nsamples * k_size;
		sum.x /= n_k_size;
		sum.y /= n_k_size;
		input[index] = sum;
		index += blockDim.x * gridDim.x;
	}
}

void convolution_kernel(cuComplex		*input,
						cuComplex		*gpu_special_queue,
						const uint		frame_resolution,
						const uint		frame_width,
						const float		*kernel,
						const uint		k_width,
						const uint		k_height,
						const uint		k_z,
						uint&			gpu_special_queue_start_index,
						const uint&		gpu_special_queue_max_index,
						cudaStream_t	stream)
{

	uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_resolution, threads);


	cudaStreamSynchronize(stream);

	if (gpu_special_queue_start_index == 0)
		gpu_special_queue_start_index = gpu_special_queue_max_index - 1;
	else
		--gpu_special_queue_start_index;
	cudaMemcpy(
		gpu_special_queue + frame_resolution * gpu_special_queue_start_index,
		input,
		sizeof(cuComplex) * frame_resolution,
		cudaMemcpyDeviceToDevice);
	
	uint gpu_special_queue_buffer_length = gpu_special_queue_max_index * frame_resolution;

	kernel_multiply_kernel<<<blocks, threads, 0, stream>>>(
		input,
		gpu_special_queue,
		gpu_special_queue_buffer_length,
		frame_resolution,
		frame_width,
		kernel,
		k_width,
		k_height,
		k_z,
		gpu_special_queue_start_index,
		gpu_special_queue_max_index);
		
	cudaStreamSynchronize(stream);
}
