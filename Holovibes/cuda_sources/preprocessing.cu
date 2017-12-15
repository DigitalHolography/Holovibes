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

#include "preprocessing.cuh"
#include "tools_conversion.cuh"

using camera::FrameDescriptor;
using holovibes::Queue;

void make_contiguous_complex(Queue&			input,
	cuComplex*		output,
	cudaStream_t	stream)
{
	const uint				threads = get_max_threads_1d();
	const uint				blocks = map_blocks_to_problem(input.get_pixels(), threads);
	const uint				frame_resolution = input.get_pixels();
	const FrameDescriptor&	frame_desc = input.get_frame_desc();

	if (frame_desc.depth == 1)
		img8_to_complex << <blocks, threads, 0, stream >> > (
			output,
			static_cast<uchar*>(input.get_start()),
			frame_resolution);
	else if (frame_desc.depth == 2)
		img16_to_complex << <blocks, threads, 0, stream >> > (
			output,
			static_cast<ushort*>(input.get_start()),
			frame_resolution);
	else if (frame_desc.depth == 4)
		float_to_complex << <blocks, threads, 0, stream >> > (
			output,
			static_cast<float*>(input.get_start()),
			frame_resolution);
	else if (frame_desc.depth == 8)
		cudaMemcpy(output, input.get_start(), frame_resolution << 3, cudaMemcpyDeviceToDevice);
	cudaCheckError();
}
