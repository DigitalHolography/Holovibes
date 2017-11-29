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

#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "queue.hh"

namespace holovibes
{
	class ComputeDescriptor;
	namespace compute
	{
		using uint = unsigned int;

		class Converts
		{
		public:
			Converts(FnVector& fn_vect,
				cuComplex* const& gpu_input_buffer,
				float* const& gpu_float_buffer,
				cufftComplex* const& gpu_stft_buffer,
				void* const& gpu_output_buffer,
				Queue* const& gpu_3d_vision,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& input_fd);

			void insert_to_float();

		private:

			void insert_to_modulus();
			void insert_to_modulus_vision3d();
			void insert_to_squaredmodulus();
			void insert_to_composite();
			void insert_to_complex();

			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			cuComplex* const& gpu_input_buffer_;
			cufftComplex* const& gpu_stft_buffer_;
			float* const& gpu_float_buffer_;
			void* const& gpu_output_buffer_;
			Queue* const& gpu_3d_vision_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
			/// }
		};
	}
}
