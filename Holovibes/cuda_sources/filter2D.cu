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

#include "filter2D.cuh"

using camera::FrameDescriptor;

__global__
void filter2D_roi(cuComplex	*input,
				const uint	tl_x,
				const uint	tl_y,
				const uint	br_x,
				const uint	br_y,
				const uint	width,
				const uint	size)
{
	const uint index = blockIdx.x * blockDim.x + threadIdx.x;

	// In ROI
	if (index < size)
	{
		uint mod_index = index % width;
		if (!(index >= tl_y * width && index < br_y * width
			&& mod_index >= tl_x && mod_index < br_x))
		{
			input[index] = make_cuComplex(0, 0);
		}
	}
}


void filter2D(cuComplex				*input,
			cuComplex				*tmp_buffer,
			const cufftHandle		plan2d,
			const holovibes::units::RectFd&	r,
			const FrameDescriptor&	desc,
			cudaStream_t			stream)
{
	uint threads = THREADS_128;
	uint blocks = map_blocks_to_problem(desc.frame_res(), threads);
	uint size = desc.width * desc.height;

	cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);

	if (!r.area())
		return;
	//int center_x = (r.x + r.bottom_right.x) >> 1;
	//int center_y = (r.top_left.y + r.bottom_right.y) >> 1;
	
	filter2D_roi << <blocks, threads, 0, stream >> >(
		input,
		r.x(),
		r.y(),
		r.bottomRight().x(),
		r.bottomRight().y(),
		desc.width,
		desc.width * desc.height);

	cudaMemcpy(tmp_buffer, input, size * sizeof (cuComplex), cudaMemcpyDeviceToDevice);

	circ_shift << <blocks, threads, 0, stream >> >(
		tmp_buffer,
		input,
		r.center().x(),
		r.center().y(),
		desc.width,
		desc.height,
		size);

	cufftExecC2C(plan2d, input, input, CUFFT_INVERSE);
}
