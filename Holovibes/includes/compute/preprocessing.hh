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

 Implementation of preprocessing features on complex buffer. */
#pragma once

#include <atomic>

#include <cufft.h>

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

			/*! \brief Describing the Ref diff/sliding state */
			enum ref_state
			{
				ENQUEUE,
				COMPUTE
			};

			/** \brief Constructor.
			
			*/
			Preprocessing(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd);

			/** \brief Allocates the ref queue.
			
			*/
			void allocate_ref(std::atomic<bool>& update_request);

			/** \brief Add normalization to all of the frames 
			
			*/
			void insert_frame_normalization();

			/** \brief Insert the interpolation function.
			
			*/
			void insert_interpolation();

			/** \brief Insert the functions relative to the Ref algorithm.
			
			*/
			void insert_ref();

			/** \brief Shifts the corners of the image

			 */
			void insert_pre_fft_shift();

			/*
				\brief Compute the intensity of an image.
			*/
			float compute_current_intensity(cufftComplex* buffer_ptr, size_t res);
		private:
			/** \brief Insert the Ref diff function.
			
			*/
			void handle_reference();
			/** \brief Insert the Ref sliding function.
			
			*/
			void handle_sliding_reference();

			//! State of the reference process
			enum ref_state					ref_diff_state_;
			//! Counter used by the reference process.
			unsigned int					ref_diff_counter_;
			//! Ref Queue. Accumulating complex images to handle reference.
			std::unique_ptr<Queue>			gpu_ref_diff_queue_;


			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			//! Main buffers
			const CoreBuffers&				buffers_;

			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;

			//! Compute Descriptor
			ComputeDescriptor&				cd_;
		};
	}
}
