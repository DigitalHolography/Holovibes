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
	struct CoreBuffers;
	struct Stft_env;
	struct UnwrappingResources;
	struct UnwrappingResources_2d;
	namespace compute
	{
		using uint = unsigned int;

		class Converts
		{
		public:
			Converts(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const Stft_env& stft_env,
				const std::unique_ptr<Queue>& gpu_3d_vision,
				const cufftHandle& plan2d,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& input_fd,
				const camera::FrameDescriptor& output_fd);

			void insert_to_float(bool unwrap_2d_requested);
			void insert_to_ushort();

		private:

			void insert_to_modulus();
			void insert_to_modulus_vision3d();
			void insert_to_squaredmodulus();
			void insert_to_composite();
			void insert_to_complex();
			void insert_to_argument(bool unwrap_2d_requested);
			void insert_to_phase_increase(bool unwrap_2d_requested);
			void insert_main_ushort();
			void insert_slice_ushort();

			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			const CoreBuffers&				buffers_;
			const Stft_env&					stft_env_;
			std::unique_ptr<UnwrappingResources>	unwrap_res_;
			std::unique_ptr<UnwrappingResources_2d>	unwrap_res_2d_;
			const std::unique_ptr<Queue>&	gpu_3d_vision_;
			const cufftHandle&				plan2d_;
			/// Describes the input frame size
			const camera::FrameDescriptor&		fd_;
			/// Describes the output frame size
			const camera::FrameDescriptor&		output_fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
			/// }
		};
	}
}
