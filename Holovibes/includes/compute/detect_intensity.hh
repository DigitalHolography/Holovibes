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

/*! \file
 *
 * Contains functions to detect intensity jumps. */
#pragma once

# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "queue.hh"
# include "rect.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct CoreBuffers;
	namespace compute
	{

		/*! \class Stabilization
		**
		*/
		class DetectIntensity
		{
		public:

			DetectIntensity(FnVector & fn_vect,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor & fd,
				ComputeDescriptor & cd);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called first
			*/
			void insert_post_contiguous_complex();

		private:

			void check_jump();
			bool can_skip_detection();
			bool is_jump(float current, float last);
			float get_current_intensity();
			void update_shift();
			void on_jump(bool delayed = false);
			void update_lambda();

			float last_intensity_;

			uint current_shift_;
			bool is_delaying_shift_;

			uint frames_since_jump_;

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
