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

  Implmentation of the conversions between buffers.*/
#pragma once

#include <memory>

#include <cufft.h>

#include "compute_descriptor.hh"
#include "frame_desc.hh"
#include "pipeline_utils.hh"
#include "queue.hh"
#include "cuda_tools\cufft_handle.hh"

namespace holovibes
{
	class ComputeDescriptor;
	struct CoreBuffers;
	struct Stft_env;
	struct UnwrappingResources;
	struct UnwrappingResources_2d;
	namespace compute
	{
		class Converts
		{
		public:
			/** \brief Constructor.

			*/
			Converts(FnVector& fn_vect,
				const CoreBuffers& buffers,
				const Stft_env& stft_env,
				cuda_tools::CufftHandle& plan2d,
				ComputeDescriptor& cd,
				const camera::FrameDescriptor& input_fd,
				const camera::FrameDescriptor& output_fd);

			/** \brief Insert functions relative to the convertion Complex => Float

			*/
			void insert_to_float(bool unwrap_2d_requested);

			/** \brief Insert functions relative to the convertion Float => Unsigned Short

			*/
			void insert_to_ushort();

			/*! \brief Insert the conversion Uint(8/16/32) => Complex frame by frame
			** 
			*/
			void insert_complex_conversion(Queue& input);

		private:

			/** \brief Set pmin_ and pmax_ according to p accumulation.

			*/
			void insert_compute_p_accu();
			/** \brief Insert the convertion Complex => Modulus

			*/
			
			void insert_to_modulus();
			
			/** \brief Insert the convertion Complex => Squared Modulus

			*/
			void insert_to_squaredmodulus();
			/** \brief Insert the convertion Complex => Composite

			*/
			void insert_to_composite();
			/** \brief Insert the convertion Complex => Argument

			*/
			void insert_to_argument(bool unwrap_2d_requested);
			/** \brief Insert the convertion Complex => Phase increase

			*/
			void insert_to_phase_increase(bool unwrap_2d_requested);
			/** \brief Insert the convertion Float => Unsigned Short in XY window

			*/
			void insert_main_ushort();
			/** \brief Insert the convertion Float => Unsigned Short in slices.

			*/
			void insert_slice_ushort();

			//! pindex.
			unsigned short pmin_;
			//! maximum value of p accumulation
			unsigned short pmax_;

			/// Vector function in which we insert the processing
			FnVector&						fn_vect_;

			//! Main buffers
			const CoreBuffers&				buffers_;
			//! STFT environment
			const Stft_env&					stft_env_;
			//! Phase unwrapping 1D. Used for phase increase and Argument.
			std::unique_ptr<UnwrappingResources>	unwrap_res_;
			//! Phase unwrapping 2D. Used for phase increase and Argument.
			std::unique_ptr<UnwrappingResources_2d>	unwrap_res_2d_;
			//! Plan 2D. Used for unwrapping.
			cuda_tools::CufftHandle&				plan2d_;
			/// Describes the input frame size
			const camera::FrameDescriptor&		fd_;
			/// Describes the output frame size
			const camera::FrameDescriptor&		output_fd_;
			/// Variables needed for the computation in the pipe
			ComputeDescriptor&				cd_;
		};
	}
}
