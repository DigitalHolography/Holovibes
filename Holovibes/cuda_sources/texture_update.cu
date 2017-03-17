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
static void kernelTextureUpdate(float				*frame,
								cudaSurfaceObject_t cuSurface,
								dim3				texDim)
{
	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// Red Mode
	//surf2Dwrite(static_cast<unsigned char>(frame[y * texDim.x + x] >> 8), cuSurface, x * 4, y);

	// Grey Mode
	if (frame[y * texDim.x + x] > 65535.f)
		frame[y * texDim.x + x] = 65535;
	else if (frame[y * texDim.x + x] < 0.f)
		frame[y * texDim.x + x] = 0;

	const uchar p = static_cast<uchar>(frame[y * texDim.x + x] / 256);
	surf2Dwrite(make_uchar4(p, p, p, 0xff), cuSurface, x << 2, y);
}

void textureUpdate(	cudaSurfaceObject_t cuSurface,
					void				*frame,
					ushort				width,
					ushort				height)
{
	dim3 threads(32, 32);
	dim3 blocks(width >> 5, height >> 5); // >> 5 == /= 32

	kernelTextureUpdate <<< blocks, threads >>>(reinterpret_cast<float *>(frame), cuSurface, dim3(width, height));
}
