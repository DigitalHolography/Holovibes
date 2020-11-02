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

#include <algorithm>
# include "texture_update.cuh"

__global__
static void updateFloatSlice(ushort* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;
	const uint index = y * texDim.x + x;

	surf2Dwrite(static_cast<uchar>(frame[index] >> 8), cuSurface, x << 2, y);
}

__global__
static void updateComplexSlice(cuComplex* frame, cudaSurfaceObject_t cuSurface, dim3 texDim)
{
	const uint xId = blockIdx.x * blockDim.x + threadIdx.x;
	const uint yId = blockIdx.y * blockDim.y + threadIdx.y;
	const uint index = yId * texDim.x + xId;


	if (frame[index].x > 65535.f)
		frame[index].x = 65535.f;
	else if (frame[index].x < 0.f)
		frame[index].x = 0.f;

	if (frame[index].y > 65535.f)
		frame[index].y = 65535.f;
	else if (frame[index].y < 0.f)
		frame[index].y = 0.f;
	float pix = hypotf(frame[index].x, frame[index].y);

	surf2Dwrite(pix, cuSurface, xId << 2, yId);
}

void textureUpdate(cudaSurfaceObject_t	cuSurface,
				void					*frame,
				const camera::FrameDescriptor&	fd,
				cudaStream_t			stream)
{

	const uint fd_width_div_32 = std::max(1u, (unsigned)fd.width / 32u);
	const uint fd_height_div_32 = std::max(1u, (unsigned)fd.height / 32u);
	dim3 blocks(fd_width_div_32, fd_height_div_32);

	unsigned thread_width = std::min(32u, (unsigned)fd.width);
	unsigned thread_height = std::min(32u, (unsigned)fd.height);
	dim3 threads(thread_width, thread_height);

	if (fd.depth == 8)
	{
		updateComplexSlice << < blocks, threads, 0, stream >> > (
			reinterpret_cast<cuComplex *>(frame),
			cuSurface,
			dim3(fd.width, fd.height));
	}
	else
	{
		updateFloatSlice << < blocks, threads, 0, stream >> > (
			reinterpret_cast<ushort *>(frame),
			cuSurface,
			dim3(fd.width, fd.height));
	}

	cudaStreamSynchronize(stream);
	cudaCheckError();
}
