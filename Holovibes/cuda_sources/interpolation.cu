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
	cudaCheckError();

	cudaUnbindTexture(comptex);
	cudaFree(tex_data);
}

