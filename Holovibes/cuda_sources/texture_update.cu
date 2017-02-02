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

__global__ void kernelTextureUpdate(cudaSurfaceObject_t cuSurface, dim3 texDim)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	//unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < texDim.x)// && y < texDim.y)
	{
		uchar4  data = make_uchar4(0xff, 0x00, 0x00, 0xff);
		//surf2Dwrite(data, cuSurface, x * sizeof(uchar4), y);
		surf1Dwrite(data, cuSurface, x * sizeof(uchar4));
	}
}

void textureUpdate(	cudaSurfaceObject_t cuSurface,
					void *frame,
					unsigned short width,
					unsigned short height)
{
	unsigned int threads = get_max_threads_1d();
	unsigned int blocks = map_blocks_to_problem(width * height, threads);

	//dim3 threads(30, 30);
	//dim3 blocks(tex->getWidth() / threads.x, tex->getHeight() / threads.y);

	kernelTextureUpdate <<< blocks, threads >>>(cuSurface, dim3(width, height));
}
