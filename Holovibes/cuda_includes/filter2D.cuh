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

#pragma once

# include "Common.cuh"

/*! \brief Function handling the stft algorithm which steps are \n

*/
void filter2D(cuComplex				*input,
			cuComplex				*tmp_buffer,
			const cufftHandle		plan2d,
			const holovibes::units::RectFd&	r,
			const camera::FrameDescriptor&	fd,
			const bool				exclude_roi,
			const bool				shift_enabled,
			cudaStream_t			stream = 0);

void filter2D_BandPass(cuComplex				*input,
					   cuComplex				*tmp_buffer,
					   const cufftHandle		plan2d,
					   const holovibes::units::RectFd&	zone,
					   const holovibes::units::RectFd& subzone,
					   const camera::FrameDescriptor&	desc,
					   const bool						shift_enabled,
					   cudaStream_t			stream = 0);