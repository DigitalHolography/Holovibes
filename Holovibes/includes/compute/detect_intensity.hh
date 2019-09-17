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

		/*! \class DetectIntensity
		**
		*/
		class DetectIntensity
		{
		public:

			/** \brief Contructor.
			
			*/
			DetectIntensity(FnVector & fn_vect,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor & fd,
				ComputeDescriptor & cd);

			/*! \brief Enqueue the appropriate functions
			**
			** Should be called at the beginning of the pipe
			*/
			void insert_post_contiguous_complex();

		private:

			/** \brief Check if there is a jump of intensity.
			 *
			 */
			void check_jump();
			/** \brief Check if we have to detect a jump of intensity.
			
			*/
			bool can_skip_detection();
			/** \brief Check if there is a jump between two intensity, according to the interpolation sensitivity.
			
			*/

			bool is_jump(float current, float last);
			/** \brief Computes the intensity of the current frame.

			*/
			float get_current_intensity();
			void update_shift();
			void on_jump(bool delayed = false);

			/** \brief Update the current interpolation wave length.
			
			*/
			void update_lambda();

			//! Intensity of the previous frame
			float last_intensity_;

			unsigned int current_shift_;
			bool is_delaying_shift_;

			//! Number of frame passed since last jump.
			unsigned int frames_since_jump_;

			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;
			//! Main buffers.
			const CoreBuffers&				buffers_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			/// Compute Descriptor
			ComputeDescriptor&				cd_;
		};
	}
}
