/*! \file
*
* Functions that will be used to compute the stft.
*/
#pragma once

# include <cuda_runtime.h>
# include <cufft.h>
#include  <device_launch_parameters.h>

/* Forward declarations. */
namespace holovibes
{
	class Rectangle;
}
namespace camera
{
	struct FrameDescriptor;
}

/*! \brief Function handling the stft algorithm which steps are \n

*/
void filter2D(
	cufftComplex*                   input,
	cufftComplex*                   tmp_buffer,
	const cufftHandle               plan2d,
	const holovibes::Rectangle&     r,
	const camera::FrameDescriptor&  desc,
	cudaStream_t stream = 0);


