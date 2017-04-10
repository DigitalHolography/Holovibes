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

#include "flowgraphy.cuh"

__global__
void kernel_flowgraphy(cuComplex	*input,
					const cuComplex	*gpu_special_queue,
					const uint		gpu_special_queue_buffer_length,
					const cuComplex	*gpu_special_queue_end,
					const uint		start_index,
					const uint		max_index,
					const uint		frame_resolution,
					const uint		i_width,
					const uint		nsamples,
					const uint		n_i)
{
	uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	cuComplex M = make_cuComplex(0, 0);
	cuComplex D = make_cuComplex(0, 0);
	//uint size = frame_resolution * nsamples;
	while (index < frame_resolution)
	{
		int deplacement = (index  + (1 + i_width + ((1 + start_index) % max_index) *  frame_resolution) * (nsamples >> 1)) % gpu_special_queue_buffer_length;
		cuComplex b = gpu_special_queue[deplacement];

		for (int k = 0; k < nsamples; ++k)
		for (int j = 0; j < nsamples; ++j)
		for (int i = 0; i < nsamples; ++i)
		{
			deplacement = (index + i + (j * i_width) + (((k + start_index) % max_index) * frame_resolution)) % gpu_special_queue_buffer_length; // while x while y, on peut virer le modulo
			cuComplex a = gpu_special_queue[deplacement];
			M.x += a.x;
			M.y += a.y;
			float diffx = a.x - b.x;
		    float diffy = a.y - b.y;
			D.x += std::sqrt(diffx * diffx + diffy * diffy); 
		}
		M.x += (n_i * b.x);
		M.y += (n_i * b.y);
		M.x /= D.x;
		M.y /= D.x;
		//M.x = pow(M.x, 2) + pow(M.y, 2);
		input[index] = M;
		index += blockDim.x * gridDim.x;
	}
}


void convolution_flowgraphy(cuComplex	*input,
						cuComplex		*gpu_special_queue,
						uint&			gpu_special_queue_start_index,
						const uint		gpu_special_queue_max_index,
						const uint		frame_resolution,
						const uint		frame_width,
						const uint		nframes,
						cudaStream_t	stream)
{
	// const uint n_frame_resolution = frame_resolution * nframes;
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

	uint n = static_cast<uint>(nframes * nframes * nframes - 3);
	uint  gpu_special_queue_buffer_length = gpu_special_queue_max_index * frame_resolution;
	cuComplex* gpu_special_queue_end = gpu_special_queue + gpu_special_queue_buffer_length;

	kernel_flowgraphy <<<blocks, threads, 0, stream>>>(	input,
														gpu_special_queue,
														gpu_special_queue_buffer_length,
														gpu_special_queue_end,
														gpu_special_queue_start_index,
														gpu_special_queue_max_index,
														frame_resolution,
														frame_width,
														nframes,
														n);

	cudaStreamSynchronize(stream);
}
