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
# define SAMPLING_FREQUENCY  0.001f

// this should avoid multiple call of cudamalloc and cudafree

static float* omega_s_mem_pool = nullptr;


static size_t omega_s_mem_pool_size = 0;


__global__
void kernel_compute_hsv()
{

}

__global__
void kernel_fill_part_frequency_axis(size_t min, size_t max, float step, float origin, float* arr)
{
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (min + id < max)
	{
		arr[min + id] = origin + id * step;
		printf("arr[%u] = %f \n",min + id , arr[min + id]);
	}
}



void hsv(cuComplex	*input,
	float *output,
	const uint frame_res,
	const ushort index_min,
	const ushort index_max,
	const float h,
	const float s,
	const float v)
{
	const uint threads = get_max_threads_1d();
	uint blocks = map_blocks_to_problem(frame_res, threads);

	size_t nb_img = std::abs(index_max - index_min) + 1;
	if (omega_s_mem_pool_size != nb_img)
	{
		omega_s_mem_pool_size = nb_img;
		if (omega_s_mem_pool)
		{ 
			cudaFree(omega_s_mem_pool);
			cudaCheckError();
		}
		
		cudaMalloc(&omega_s_mem_pool, sizeof(float) * nb_img * 2); // w1[] && w2[]
		cudaCheckError();
	

	float step = SAMPLING_FREQUENCY / nb_img;
	size_t after_mid_index = nb_img / 2 + 1;
	printf("For %u image(s) the array is \n", nb_img);
	cudaStreamSynchronize(0);
	kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(0, after_mid_index, step, 0, omega_s_mem_pool);
	cudaStreamSynchronize(0);
	float negative_origin = -SAMPLING_FREQUENCY / 2;
	
	if(nb_img % 2)
		negative_origin += step / 2;
	else
		negative_origin += step;
		//after_mid_index--; // check DEBUG
	
	kernel_fill_part_frequency_axis <<<blocks, threads, 0, 0 >>>(after_mid_index, nb_img, step, negative_origin, omega_s_mem_pool);
	cudaStreamSynchronize(0);
	printf("ENDD\n");
	}
	
	
}