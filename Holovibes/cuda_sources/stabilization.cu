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

#include "stabilization.cuh"
#include "common.cuh"
#include "cuda_tools/unique_ptr.hh"

#include <stdio.h>

struct rect
{
	int x;
	int y;
	int w;
	int h;
};

struct point
{
	int x;
	int y;
};

__global__
void kernel_extract_frame(const float	*input,
						float			*output,
						const uint		input_w,
						const struct rect	frame)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / frame.w;
	if (new_y < frame.h)
	{
		const uint new_x = index % frame.w;
		const uint old_x = new_x + frame.x;
		const uint old_y = new_y + frame.y;

		const uint old_index = old_y * input_w + old_x;
		output[index] = input[old_index];
	}
}

void extract_frame(const float	*input,
				float			*output,
				const uint		input_w,
				const holovibes::units::RectFd&	frame)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(frame.area(), threads);
	struct rect rect_frame = { frame.x(), frame.y(), frame.unsigned_width(), frame.unsigned_height() };
	kernel_extract_frame << <blocks, threads, 0, 0 >> > (input, output, input_w, rect_frame);
	cudaCheckError();
	cudaStreamSynchronize(0);
}

__global__
void kernel_resize(const float	*input,
					float		*output,
					const struct point old_size,
					const struct point new_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / new_size.x;
	if (new_y < new_size.y)
	{
		const uint new_x = index % new_size.x;
		const uint old_x = new_x / (new_size.x / old_size.x);
		const uint old_y = new_y / (new_size.y / old_size.y);

		const uint old_index = old_y * old_size.x + old_x;
		output[index] = input[old_index];
	}
}


void gpu_resize(const float		*input,
				float			*output,
				QPoint			old_size,
				QPoint			new_size,
				cudaStream_t	stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(new_size.x() * new_size.y(), threads);
	struct point old_s = { old_size.x(), old_size.y() };
	struct point new_s = { new_size.x(), new_size.y() };
	kernel_resize << <blocks, threads, 0, stream >> > (input, output, old_s, new_s);
	cudaCheckError();
}


template <typename T>
__global__
void kernel_rotation_180(T				*frame,
						point			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint new_y = index / size.x;
	if (new_y < size.y / 2)
	{
		const uint new_x = index % size.y;
		const uint old_y = size.y - new_y - 1;
		const uint old_x = size.x - new_x - 1;
		const uint old_index = old_y * size.x + old_x;
		T tmp = frame[old_index];
		frame[old_index] = frame[index];
		frame[index] = tmp;
	}
}

void rotation_180(float			*frame,
					QPoint		size,
					cudaStream_t stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_rotation_180 << <blocks, threads, 0, stream >> > (frame, s);
	cudaCheckError();
}

void rotation_180(cuComplex		*frame,
					QPoint		size,
					cudaStream_t stream)
{
	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(size.x() * size.y(), threads);
	struct point s = { size.x(), size.y() };
	kernel_rotation_180 << <blocks, threads, 0, stream >> > (frame, s);
	cudaCheckError();
}








