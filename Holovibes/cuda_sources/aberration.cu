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

#include "aberration.cuh"
#include "unique_ptr.hh"
# include "tools_compute.cuh"
# include "cufft_handle.hh"
#include "operator_overload.cuh"


static __device__ cuComplex phi(cuComplex shift, uint x, uint y)
{
	uint tmp = shift.x * x + shift.y * y;
	cuComplex res;
	res.x = cosf(tmp);
	res.y = sinf(tmp);
	return res;
}

static __global__
void kernel_apply_aberration_phis(cufftComplex*			lens,
								  const cufftComplex*	shifts,
								  const uint			frame_size,
								  const uint			frame_width,
								  const uint			chunks_per_row,
								  const uint			chunk_width,
								  const uint			chunk_height)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < frame_size)
	{
		const uint x = index % frame_width;
		const uint y = index / frame_width;

		const uint chunk_y = y / chunk_height;
		const uint chunk_x = x / chunk_width;

		const uint chunk_no = chunk_y * chunks_per_row + chunk_x;

		lens[index].x = 32000;		//Just to test
		lens[index].y = 32000;

		lens[index] = lens[index] * phi(shifts[chunk_no], x, y);
	}
}

void apply_aberration_phis(ComplexArray& lens,
						   const std::vector<cufftComplex>& shifts,
						   unsigned int nb_chunks_per_row,
						   unsigned int nb_chunks_per_column,
						   const camera::FrameDescriptor& fd)
{
	holovibes::cuda_tools::UniquePtr<cufftComplex> gpu_shifts(shifts.size());
	cudaMemcpy(gpu_shifts, shifts.data(), shifts.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(fd.frame_res(), threads);
	kernel_apply_aberration_phis << <threads, blocks, 0, 0 >> >
		(lens,
		gpu_shifts,
		fd.frame_res(),
		fd.width,
		nb_chunks_per_row,
		fd.width / nb_chunks_per_row,
		fd.height / nb_chunks_per_column);
	cudaStreamSynchronize(0);
	cudaCheckError();
}