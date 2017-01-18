/*! \file
*
* Functions that will be used to compute the stft.
*/
#pragma once

# include "cuda_shared.cuh"
# include  <device_launch_parameters.h>
# include "tools.cuh"

/*! \brief Function handling the stft algorithm which steps are \n

*/
void filter2D(	complex							*input,
				complex							*tmp_buffer,
				const cufftHandle				plan2d,
				const holovibes::Rectangle&		r,
				const camera::FrameDescriptor&	desc,
				cudaStream_t					stream = 0);