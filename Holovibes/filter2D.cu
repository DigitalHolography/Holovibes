
#include "filter2D.cuh"
#include "hardware_limits.hh"
#include "geometry.hh"
#include "frame_desc.hh"
#include "tools.hh"

__global__ void filter2D_roi(
	cufftComplex *input,
	const unsigned int tl_x,
	const unsigned int tl_y,
	const unsigned int br_x,
	const unsigned int br_y,
	const unsigned int width,
	const unsigned int size)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	// In ROI
	while (index < size)
	{
		unsigned int mod_index = index % width;
		if (!(index >= tl_y * width && index < br_y * width
			&& mod_index >= tl_x && mod_index < br_x))
		{
			input[index] = make_cuComplex(0, 0);
		}
		index += blockDim.x * gridDim.x;
	}
}


void filter2D(
	cufftComplex*                   input,
	cufftComplex*                   tmp_buffer,
	const cufftHandle               plan2d,
	const holovibes::Rectangle&     r,
	const camera::FrameDescriptor&  desc,
	cudaStream_t stream)
{
	unsigned int threads = 128;
	unsigned int blocks = map_blocks_to_problem(desc.frame_res(), threads);
	unsigned int size = desc.width * desc.height;
	
	cufftExecC2C(plan2d, input, input, CUFFT_FORWARD);
	cudaStreamSynchronize(stream);

	if (!r.area())
		return;
	int center_x = (r.top_left.x + r.bottom_right.x) >> 1;
	int center_y = (r.top_left.y + r.bottom_right.y) >> 1;
	
	filter2D_roi << <blocks, threads, 0, stream >> >(
		input,
		r.top_left.x,
		r.top_left.y,
		r.bottom_right.x,
		r.bottom_right.y,
		desc.width,
		desc.width * desc.height);

	cudaMemcpy(tmp_buffer, input, size * sizeof (cufftComplex), cudaMemcpyDeviceToDevice);

	circ_shift << <blocks, threads, 0, stream >> >(
		tmp_buffer,
		input,
		center_x,
		center_y,
		desc.width,
		desc.height,
		size);

	cufftExecC2C(plan2d, input, input, CUFFT_INVERSE);
}