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


# include <cufft.h>
# include <qglobal.h>
# include <cuComplex.h>
# include "cuda_tools/unique_ptr.hh"
# include "cuda_tools/array.hh"
# include "pipeline_utils.hh"
# include "frame_desc.hh"
# include "queue.hh"
# include "rect.hh"

namespace holovibes
{
	class ComputeDescriptor;
	namespace compute
	{

		/*! \class Stabilization
		**
		*/
		class DetectIntensity
		{
		public:
			DetectIntensity(FnVector& fn_vect,
				cuComplex* const& gpu_input_buffer,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called first
			*/
			void insert_post_contiguous_complex();

		private:

			void check_jump();
			bool is_jump(float current, float last);
			float get_current_intensity();
			void on_jump();
			void update_lambda();

			float last_intensity_;

			uint frames_since_jump_;
			uint sum_frames_;
			uint nb_jumps_;

			/// Pipe data
			/// {
			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			/// The whole image for this frame
			cuComplex* const&				gpu_input_buffer_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;

			ComputeDescriptor&				cd_;
			/// }
		};
	}
}
