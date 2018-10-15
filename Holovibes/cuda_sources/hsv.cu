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

# include "hsv.cuh"
# include <stdio.h>
# define SAMPLING_FREQUENCY  1


/*
	\brief simple memory pool which should avoid multiple call of cudamalloc and cudafree
	data is made of w1 and w2
	size is the number of images taken simultaneously.
	The size of data is (size * 2 * sizeof(float))
*/

struct omega_s_mem_pool {
	float* data;
	size_t size;
};

static struct omega_s_mem_pool omega_arr = {nullptr, 0};

/*
** \brief Compute H component of hsv.
** h is separated from s and v in order not use an array of 1's
** 
*/
__global__
void kernel_compute_and_fill_h(const cuComplex* input, float* output, const size_t frame_res, const size_t min_index, const size_t max_index)
{
	// if you find how to remove the loop do it :)
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id] = 0;
		for (size_t i = min_index; i <= max_index; ++i)
		{
			output[id] += input[i * frame_res + id].x;
		}
		output[id] /= (max_index - min_index + 1);
	}
}

__global__
void kernel_compute_and_fill_s(const cuComplex* input, float* output, const size_t frame_res, const size_t min_index, const size_t max_index, const float* omega_s_arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id + frame_res] = 0;
		for (size_t i = min_index; i <= max_index; ++i)
		{
			output[id + frame_res] += input[i * frame_res + id].x * omega_s_arr[i];
		}
		output[id + frame_res] /= (max_index - min_index + 1);
	}
}

__global__
void kernel_compute_and_fill_v(const cuComplex* input, float* output, const size_t frame_res, const size_t min_index, const size_t max_index, const omega_s_mem_pool* omega)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < frame_res)
	{
		output[id + 2 * frame_res] = 0;
		for (size_t i = min_index; i <= max_index; ++i)
		{
			output[id + 2 * frame_res] += input[i * frame_res + id].x * omega->data[omega->size + i];
		}
		output[id + 2 * frame_res] /= (max_index - min_index + 1);
	}
}




__global__
void kernel_fill_square_frenquency_axis(const size_t length, float* arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < length)
	{
		arr[length + id] = arr[id] * arr[id];
	}
}

__global__
void kernel_fill_part_frequency_axis(const size_t min, const size_t max, const double step, const double origin, float* arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (min + id < max)
	{
		arr[min + id] = origin + id * step;
		printf("arr[%u] = %f \n",min + id , arr[min + id]);
	}
}



void hsv(const cuComplex	*input,
	float *output,
	const uint frame_res,
	const ushort index_min,
	const ushort index_max,
	const uint nb_img,
	const float h,
	const float s,
	const float v)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_res, threads);

	if (omega_arr.size != nb_img)
	{
		omega_arr.size = nb_img;
		if (omega_arr.data)
		{ 
			cudaFree(omega_arr.data);
			cudaCheckError();
		}
		
		cudaMalloc(&omega_arr.data, sizeof(float) * nb_img * 2); // w1[] && w2[]
		cudaCheckError();
	
		double step = SAMPLING_FREQUENCY / (double)nb_img;
		size_t after_mid_index = nb_img / (double)2.0 + (double)1.0;
		kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(0, after_mid_index, step, 0, omega_arr.data);
		double negative_origin = -SAMPLING_FREQUENCY / (double)2.0;
		if(nb_img % 2)
			negative_origin += step / (double)2.0;
		else 
			negative_origin += step;
		kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(after_mid_index, nb_img, step, negative_origin, omega_arr.data);
		cudaCheckError();
		kernel_fill_square_frenquency_axis <<<blocks, threads, 0, 0 >>> (nb_img, omega_arr.data);
		cudaCheckError();
	}
	

	
}