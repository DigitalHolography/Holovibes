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
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Red Mode
	unsigned short pix = frame[(y * texDim.x + x)];
	//float pix_f = pix / 65536.f * 256;
	surf2Dwrite(static_cast<unsigned short>(pix >> 8), cuSurface, x * 4, y);

	// Grey Mode
	/*unsigned short pix0 = frame[(y * texDim.x + x)];
	float pix_f = pix0 / 256;
	unsigned char p = static_cast<unsigned char>(pix_f);
	uchar4 p4 = make_uchar4(p, p, p, 0xff);

	surf2Dwrite(p4, cuSurface, x * 4, y);*/
}

void textureUpdate(	cudaSurfaceObject_t cuSurface,
					void *frame,
					unsigned short width,
					unsigned short height)
{
	dim3 threads(32, 32);
	dim3 blocks(width / threads.x, height / threads.y);

	kernelTextureUpdate << < blocks, threads >> >(reinterpret_cast<unsigned short*>(frame),
		cuSurface, dim3(width, height));
}
