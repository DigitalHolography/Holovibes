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

void make_sqrt_vect(float			*out,
					const ushort	n,
					cudaStream_t	stream)
{
	float *vect = new float[n]();

	for (size_t i = 0; i < n; ++i)
		vect[i] = sqrtf(static_cast<float>(i));

	cudaMemcpyAsync(out, vect, sizeof(float) * n, cudaMemcpyHostToDevice, stream);

	delete[] vect;
}

void make_contiguous_complex(Queue&			input,
							cuComplex*		output,
							const uint		n,
							cudaStream_t	stream)
{
	const uint				threads = get_max_threads_1d();
	const uint				blocks = map_blocks_to_problem(input.get_pixels() * n, threads);
	const uint				frame_resolution = input.get_pixels();
	const FrameDescriptor&	frame_desc = input.get_frame_desc();

	if (input.get_start_index() + n <= input.get_max_elts())
	{
		const uint n_frame_resolution = frame_resolution * n;
		/* Contiguous case. */
		if (frame_desc.depth == 2.f)
		{
			img16_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<ushort*>(input.get_start()),
				n_frame_resolution);
		}
		else if (frame_desc.depth == 1.f)
		{
			img8_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<uchar*>(input.get_start()),
				n_frame_resolution);
		}
		else if (frame_desc.depth == 4.f)
		{
			float_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<float*>(input.get_start()),
				n_frame_resolution);
		}
		else if (frame_desc.depth == 8.f)
			cudaMemcpy(output, input.get_start(), n_frame_resolution << 3, cudaMemcpyDeviceToDevice); // frame_res * 8
	}
	else
	{
		const uint contiguous_elts = input.get_max_elts() - input.get_start_index();
		const uint contiguous_elts_res = frame_resolution * contiguous_elts;
		const uint left_elts = n - contiguous_elts;
		const uint left_elts_res = frame_resolution * left_elts;

		if (frame_desc.depth == 2.f)
		{
			// Convert contiguous elements (at the end of the queue).
			img16_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<ushort*>(input.get_start()),
				contiguous_elts_res);

			// Convert the contiguous elements left (at the beginning of queue).
			img16_to_complex << <blocks, threads, 0, stream >> >(
				output + contiguous_elts_res,
				static_cast<ushort*>(input.get_buffer()),
				left_elts_res);
		}
		else if (frame_desc.depth == 1.f)
		{
			// Convert contiguous elements (at the end of the queue).
			img8_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<uchar*>(input.get_start()),
				contiguous_elts_res);

			// Convert the contiguous elements left (at the beginning of queue).
			img8_to_complex << <blocks, threads, 0, stream >> >(
				output + contiguous_elts_res,
				static_cast<uchar*>(input.get_buffer()),
				left_elts_res);
		}
		else if (frame_desc.depth == 4.f)
		{
			// Convert contiguous elements (at the end of the queue).
			float_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<float *>(input.get_start()),
				contiguous_elts_res);

			// Convert the contiguous elements left (at the beginning of queue).
			float_to_complex << <blocks, threads, 0, stream >> >(
				output + contiguous_elts_res,
				static_cast<float*>(input.get_buffer()),
				left_elts_res);
		}
		else if (frame_desc.depth == 8.f)
		{
			cudaMemcpy(output, input.get_start(), contiguous_elts_res, cudaMemcpyDeviceToDevice);
			cudaMemcpy(output + contiguous_elts_res, input.get_buffer(), left_elts_res, cudaMemcpyDeviceToDevice);
		}
	}
}
