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
#include "pipeline_utils.hh"
#include "frame_desc.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct CoreBuffers;

	namespace compute
	{
		class Preprocessing
		{
		public:

			enum ref_state
			{
				ENQUEUE,
				COMPUTE
			};

			Preprocessing(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd);

			void allocate_ref(std::atomic<bool>& update_request);
			void insert_interpolation();
			void insert_ref();
		private:
			void handle_reference();
			void handle_sliding_reference();

			enum ref_state					ref_diff_state_;
			uint							ref_diff_counter_;
			std::unique_ptr<Queue>			gpu_ref_diff_queue_;


			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			const CoreBuffers&				buffers_;

			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;

			ComputeDescriptor&				cd_;
			/// }
		};
	}
}
