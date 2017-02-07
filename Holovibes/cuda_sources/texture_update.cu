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
void kernelTextureUpdate(	unsigned short* frame,
							cudaSurfaceObject_t cuSurface,
							dim3 texDim)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Red Mode
	surf2Dwrite(static_cast<unsigned char>(frame[y * texDim.x + x] >> 8), cuSurface, x * 4, y);

	// Grey Mode
	/*const unsigned char p = static_cast<unsigned char>(frame[y * texDim.x + x] >> 8);
	surf2Dwrite(make_uchar4(p, p, p, 0xff), cuSurface, x * 4, y);*/
}

void textureUpdate(	cudaSurfaceObject_t cuSurface,
					void *frame,
					unsigned short width,
					unsigned short height)
{
	dim3 threads(32, 32);
	dim3 blocks(width >> 5, height >> 5);

	kernelTextureUpdate <<< blocks, threads >>>(reinterpret_cast<unsigned short*>(frame),
		cuSurface, dim3(width, height));
}
