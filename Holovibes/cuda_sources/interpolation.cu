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

# include "interpolation.cuh"

// global declaration of 2D complex texture (visible for host and device code)
texture<cuComplex, cudaTextureType2D, cudaReadModeElementType> comptex;

static __global__
void kernel_bilinear_tex_interpolation(cuComplex *__restrict__ output,
									const int M1,
									const int M2,
									const float ratio)
{
	const int index = threadIdx.x + blockDim.x * blockIdx.x;

	const int i = index % M1;
	const int j = index / M1;

	if (i < M1 && j < M2)
	{
		output[index] = tex2D(comptex, i / ratio + 0.5, j / ratio + 0.5);
		output[index].y = 0;
	}
}

static __global__
void kernel_bilinear_manual_interpolation(cuComplex *__restrict__ output,
									cuComplex * input,
									const int M1,
									const int M2,
									const float ratio)
{
	const int l = threadIdx.x + blockDim.x * blockIdx.x;

	const int i = l % M1;
	const int j = l / M1;

	const int N1 = M1;
	const int N2 = M2;

	if (i < N1 && j < N2)
	{
		const float	x_pos = i / ratio;
		const int ind_x = floor(x_pos);
		const float a = x_pos - ind_x;

		const float y_pos = j / ratio;
		const int ind_y = floor(y_pos);
		const float b = y_pos - ind_y;

		float d00;
		float d01;
		float d10;
		float d11;

		if (ind_x < M1 && ind_y < M2)
			d00 = input[ind_y * M1 + ind_x].x;
		else
			d00 = 0.f;

		if ((ind_x + 1) < M1 && (ind_y) < M2)
			d10 = input[ind_y * M1 + ind_x + 1].x;
		else
			d10 = 0.f;

		if (ind_x < M1 && (ind_y + 1) < M2)
			d01 = input[(ind_y + 1) * M1 + ind_x].x;
		else
			d01 = 0.f;

		if ((ind_x + 1) < M1 && (ind_y + 1) < M2)
			d11 = input[(ind_y + 1) * M1 + ind_x + 1].x;
		else
			d11 = 0.f;

		float result_temp1;
		float result_temp2;
		result_temp1 = a * d10 + (-d00 * a + d00);
		result_temp2 = a * d11 + (-d01 * a + d01);

		output[l].x = b * result_temp2 + (-result_temp1 * b + result_temp1);
		output[l].y = 0;
	}
}

void tex_interpolation(cuComplex *buffer,
				  const unsigned int width,
				  const unsigned int height,
				  const float ratio,
				  cudaStream_t stream)
{
	size_t pitch;
	cuComplex* tex_data;

	// Setting texture for linear interpolation
	comptex.addressMode[0] = cudaAddressModeWrap;
	comptex.addressMode[1] = cudaAddressModeWrap;
	comptex.filterMode = cudaFilterModeLinear;
	// Coordinates not normalized
	comptex.normalized = false;

	cudaMallocPitch((void**)&tex_data, &pitch, width * sizeof(cuComplex), height);
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();
	// Binding texture to its data
	cudaBindTexture2D(0, &comptex, tex_data, &desc, width, height, pitch);
	// Copying input into texture data
	cudaMemcpy2D(tex_data, pitch, buffer, sizeof(cuComplex) * width, sizeof(cuComplex) * width, height, cudaMemcpyHostToDevice);

	const uint threads = get_max_threads_1d();
	const uint blocks = map_blocks_to_problem(width * height, threads);

	kernel_bilinear_tex_interpolation << <blocks, threads, 0, stream >> > (buffer, width, height, ratio);

	cudaUnbindTexture(comptex);
	cudaFree(tex_data);
}

