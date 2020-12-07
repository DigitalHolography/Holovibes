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

#include "tools.cuh"
#include "tools_compute.cuh"
#include "tools_unwrap.cuh"
#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/cufft_handle.hh"
#include "logger.hh"
#include "cuda_memory.cuh"
#include "Common.cuh"

#include <cassert>

using camera::FrameDescriptor;
using namespace holovibes;
using cuda_tools::UniquePtr;
using cuda_tools::CufftHandle;

__global__
void kernel_apply_lens(cuComplex		*input,
					cuComplex 			*output,
					const uint 			batch_size,
					const uint			input_size,
					const cuComplex		*lens,
					const uint			lens_size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < input_size)
	{
		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * input_size;

			const float	tmp_x = input[batch_index].x;
			output[batch_index].x = input[batch_index].x * lens[index].x - input[batch_index].y * lens[index].y;
			output[batch_index].y = input[batch_index].y * lens[index].x + tmp_x * lens[index].y;
		}
	}
}

__global__
void kernel_complex_to_modulus(const cuComplex	*input,
							float				*output,
							const uint			size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size)
		output[index] = hypotf(input[index].x, input[index].y);
}

void frame_memcpy(const float				*input,
				const units::RectFd&	zone,
				const uint			input_width,
				float				*output,
				cudaStream_t		stream)
{
	const float	*zone_ptr = input + (zone.topLeft().y() * input_width + zone.topLeft().x());
	cudaSafeCall(cudaMemcpy2DAsync(output,
					  zone.width() * sizeof(float),
					  zone_ptr,
					  input_width * sizeof(float),
					  zone.width() * sizeof(float),
					  zone.height(),
					  cudaMemcpyDeviceToDevice,
					  stream));
	cudaStreamSynchronize(stream);
}

__global__
void circ_shift(const cuComplex	*input,
				cuComplex	*output,
				const uint 	batch_size,
				const int	i, // shift on x axis
				const int	j, // shift on y axis
				const uint	width,
				const uint	height,
				const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		int index_x = index % width;
		int index_y = index / width;
		int shift_x = index_x - i;
		int shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;

		for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * size;

			const cuComplex rhs = input[batch_index];

			output[((width * shift_y) + shift_x) + i * size] = rhs;
		}
	}
}

__global__
void circ_shift_float(const float		*input,
					float		*output,
					const uint 	batch_size,
					const int	i, // shift on x axis
					const int	j, // shift on y axis
					const uint	width,
					const uint	height,
					const uint	size)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
	{
		int index_x = index % width;
		int index_y = index / width;
		int shift_x = index_x - i;
		int shift_y = index_y - j;
		shift_x = (shift_x < 0) ? (width + shift_x) : shift_x;
		shift_y = (shift_y < 0) ? (height + shift_y) : shift_y;

        for (uint i = 0; i < batch_size; ++i)
		{
			const uint batch_index = index + i * size;

			const float rhs = input[batch_index];

			output[((width * shift_y) + shift_x) + i * size] = rhs;
		}
	}
}

__global__
void kernel_translation(float		*input,
						float		*output,
						uint		width,
						uint		height,
						int			shift_x,
						int			shift_y)
{
	const uint	index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < width * height)
	{
		const int new_x = index % width;
		const int new_y = index / width;
		const int old_x = (new_x - shift_x + width) % width;
		const int old_y = (new_y - shift_y + height) % height;
		output[index] = input[old_y * width + old_x];
	}
}

