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

 Implementation of postprocessing features on complex buffers. */
#pragma once

#include "pipeline_utils.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
	class ComputeDescriptor;
	struct CoreBuffers;

	namespace compute
	{
		class Postprocessing
		{
		public:
			/** \brief Constructor.
			
			*/
			Postprocessing(FnVector& fn_vect,
				CoreBuffers& buffers,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd);

			/** \brief Allocates convolution and flowgraphy buffers.
			
			*/
			void allocate_buffers();

			/** \brief Insert the Vibrometry function.
			
			*/
			void insert_vibrometry();
			/** \brief Insert the Convolution function. TODO: Check if it works.
			
			*/
			void insert_convolution();
			/** \brief Insert the Flowgrpahy function. TODO: FIX, it doesn't work.
			
			*/
			void insert_flowgraphy();

		private:

			//queue used for flowgraphy
			cuda_tools::UniquePtr<cufftComplex>	gpu_special_queue_;
			cuda_tools::UniquePtr<float>		gpu_kernel_buffer_;
			uint								gpu_special_queue_start_index_;
			uint								gpu_special_queue_max_index_;

			// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			//! Main buffers
			CoreBuffers&				buffers_;

			// Describes the frame size
			const camera::FrameDescriptor&	fd_;

			//! Compute Descriptor
			ComputeDescriptor&				cd_;

			// plan used for the convolution (frame width, frame height, cufft_c2c)
			CufftHandle						plan_;
		};
	}
}

