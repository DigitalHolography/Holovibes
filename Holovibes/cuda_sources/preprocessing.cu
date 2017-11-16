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

texture<unsigned short, cudaTextureType2D, cudaReadModeNormalizedFloat> shorttex;
texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> chartex;

static __global__
void kernel_bilinear_tex_short_interpolation(unsigned short *__restrict__ output,
									const int M1,
									const int M2,
									const float ratio)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;

	const int i = index % M1;
	const int j = index / M1;

	if (i < M1 && j < M2)
	{
		float val = tex2D(shorttex, i / ratio + 0.5, j / ratio + 0.5);
		output[index] = val;
	}
}

static __global__
void kernel_bilinear_tex_char_interpolation(unsigned char *__restrict__ output,
									const int M1,
									const int M2,
									const float ratio)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;

	const int i = index % M1;
	const int j = index / M1;

	if (i < M1 && j < M2)
	{
		output[index] = tex2D(chartex, i / ratio + 0.5, j / ratio + 0.5);
	}
}


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


static void short_interpolation(unsigned int width, unsigned int height, unsigned short *buffer, const float ratio, cudaStream_t stream)
{
	size_t pitch;
	size_t tex_ofs;
	unsigned short* tex_data;

	// Setting texture for linear interpolation
	shorttex.filterMode = cudaFilterModeLinear;
	// Coordinates not normalized
	shorttex.normalized = false;

	cudaMallocPitch((void**)&tex_data, &pitch, width * sizeof(unsigned short), height);
	// Copying input into texture data
	cudaMemcpy2D(tex_data, pitch, buffer, sizeof(unsigned short) * width, sizeof(unsigned short) * width, height, cudaMemcpyDeviceToDevice);
	// Binding texture to its data
	cudaBindTexture2D(&tex_ofs, &shorttex, tex_data, &shorttex.channelDesc, width, height, pitch);

	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(width * height, threads);

	kernel_bilinear_tex_short_interpolation << <blocks, threads, 0, stream >> > (buffer, width, height, ratio);

	cudaUnbindTexture(shorttex);
	cudaFree(tex_data);
}

static void char_interpolation(unsigned int width, unsigned int height, unsigned char *buffer, const float ratio, cudaStream_t stream)
{
	size_t pitch;
	size_t tex_ofs;
	unsigned char* tex_data;

	// Setting texture for linear interpolation
	chartex.filterMode = cudaFilterModeLinear;
	// Coordinates not normalized
	chartex.normalized = false;

	cudaMallocPitch((void**)&tex_data, &pitch, width * sizeof(unsigned char), height);
	// Copying input into texture data
	cudaMemcpy2D(tex_data, pitch, buffer, sizeof(unsigned char) * width, sizeof(unsigned char) * width, height, cudaMemcpyDeviceToDevice);
	// Binding texture to its data
	cudaBindTexture2D(&tex_ofs, &chartex, tex_data, &shorttex.channelDesc, width, height, pitch);

	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(width * height, threads);

	kernel_bilinear_tex_char_interpolation << <blocks, threads, 0, stream >> > (buffer, width, height, ratio);

	cudaUnbindTexture(chartex);
	cudaFree(tex_data);
}

void make_contiguous_complex(Queue&			input,
							cuComplex*		output,
							const uint		n,
							const float ratio,
							bool interpolation,
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
			if (interpolation)
				short_interpolation(
					input.get_frame_desc().width,
					input.get_frame_desc().height,
					static_cast<ushort*>(input.get_start()),
					ratio,
					stream);
			img16_to_complex << <blocks, threads, 0, stream >> >(
				output,
				static_cast<ushort*>(input.get_start()),
				n_frame_resolution);
		}
		else if (frame_desc.depth == 1.f)
		{
			if (interpolation)
				char_interpolation(
					input.get_frame_desc().width,
					input.get_frame_desc().height,
					static_cast<uchar*>(input.get_start()),
					ratio,
					stream);
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
