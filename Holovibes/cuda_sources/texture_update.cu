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

# include "tools.hh"
# include "texture_update.cuh"

__global__
static void updateSliceTexture(float* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (frame[y * texDim.x + x] > 65535.f)
		frame[y * texDim.x + x] = 65535;
	else if (frame[y * texDim.x + x] < 0.f)
		frame[y * texDim.x + x] = 0;

	const uchar p = static_cast<uchar>(frame[y * texDim.x + x] / 256.f);
	//surf2Dwrite(make_uchar4(p, p, p, 0xff), cuSurface, x << 2, y);
	surf2Dwrite(p, cuSurface, x << 2, y);
}

/*__global__
static void TextureUpdate_8bit(unsigned char* frame,
						cudaSurfaceObject_t cuSurface,
						dim3 texDim)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	surf2Dwrite(frame[y * texDim.x + x], cuSurface, x << 2, y);
}

__global__
static void TextureUpdate_16bit(unsigned short* frame,
						cudaSurfaceObject_t cuSurface,
						dim3 texDim)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	surf2Dwrite(
		static_cast<unsigned char>(frame[y * texDim.x + x] >> 8), cuSurface, x << 2, y);
}*/

void textureUpdate(cudaSurfaceObject_t		cuSurface,
					void					*frame,
					const FrameDescriptor&	fd,
					cudaStream_t			stream)
{
	//dim3 threads(32, 32);
	//dim3 blocks(fd.width >> 5, fd.height >> 5);
	uint threads_2d = get_max_threads_2d();
	dim3 lthreads(threads_2d, threads_2d);
	dim3 lblocks(fd.width / threads_2d, fd.height / threads_2d);
	updateSliceTexture << < lblocks, lthreads, 0, stream >> >(
		reinterpret_cast<float *>(frame),
		cuSurface, dim3(fd.width, fd.height));

	/*if (Fd.depth == 1)
	{
		TextureUpdate_8bit << < blocks, threads >> > (
			reinterpret_cast<unsigned char*>(frame),
			cuSurface,
			dim3(Fd.width, Fd.height));
	}
	else if (Fd.depth == 2)
	{
		TextureUpdate_16bit << < blocks, threads >> > (
			reinterpret_cast<unsigned short*>(frame),
			cuSurface,
			dim3(Fd.width, Fd.height));
	}*/
	cudaStreamSynchronize(stream);
}
