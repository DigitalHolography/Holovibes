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

#include "postprocessing.hh"
#include "icompute.hh"
#include "compute_descriptor.hh"
#include "vibrometry.cuh"
#include "convolution.cuh"
#include "flowgraphy.cuh"

namespace holovibes
{
	namespace compute
	{
		Postprocessing::Postprocessing(FnVector& fn_vect,
			const CoreBuffers& buffers,
			const camera::FrameDescriptor& input_fd,
			ComputeDescriptor& cd)
			: fn_vect_(fn_vect)
			, buffers_(buffers)
			, fd_(input_fd)
			, cd_(cd)
		{
		}

		void Postprocessing::insert_vibrometry()
		{
			if (cd_.vibrometry_enabled)
			{
				cufftComplex* qframe = buffers_.gpu_input_buffer_ + fd_.frame_res();
				fn_vect_.push_back([=]() {
					frame_ratio(
						buffers_.gpu_input_buffer_,
						qframe,
						buffers_.gpu_input_buffer_,
						fd_.frame_res());
				});
			}
		}

		void Postprocessing::insert_convolution()
		{

		}

		void Postprocessing::insert_flowgraphy()
		{

		}
	}
}