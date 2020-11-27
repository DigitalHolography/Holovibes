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

 Implementation of FFT1, FFT2 and STFT algorithms. */
#pragma once

#include <cufft.h>

#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"

namespace holovibes
{
	class Queue;
	class ComputeDescriptor;
	struct BatchEnv;
	struct TimeTransformationEnv;
	struct CoreBuffersEnv;

	namespace compute
	{
		class FourierTransform
		{
		public:
			/*! \brief Constructor.

			*/
			FourierTransform(FunctionVector& fn_compute_vect,
				const CoreBuffersEnv& buffers,
				const camera::FrameDescriptor& fd,
				holovibes::ComputeDescriptor& cd,
				cuda_tools::CufftHandle& spatial_transformation_plan,
				const BatchEnv& batch_env,
				TimeTransformationEnv& time_transformation_env);

			/*! \brief enqueue functions relative to spatial fourier transforms.

			*/
			void insert_fft();

			/*! \brief enqueue functions that store the p frame after the time transformation.

			*/
			void FourierTransform::insert_store_p_frame();

			/*! \brief Get Lens Queue used to display the Fresnel lens.

			*/
			std::unique_ptr<Queue>& get_lens_queue();

			/*! \brief enqueue functions relative to temporal fourier transforms.

			*/
			void insert_stft();

			/*! \brief Enqueue functions relative to filtering using diagonalization and eigen values.
					   This should eventually replace stft
			*/
			void insert_eigenvalue_filter();

			/*! \brief Enqueue functions relative to time transformation cuts display when there are activated

			*/
			void insert_time_transformation_cuts_view();

		private:
			/*! \brief Enqueue the call to filter2d cuda function.

			*/
			void insert_filter2d();

			/*! \brief Compute lens and enqueue the call to fft1 cuda function.

			*/
			void insert_fft1();

			/*! \brief Compute lens and enqueue the call to fft2 cuda function.

			*/
			void insert_fft2();

			/*! \brief Enqueue the Fresnel lens into the Lens Queue.

				It will enqueue the lens, and normalize it, in order to display it correctly later.
			*/
			void enqueue_lens();

			//! Roi zone of Filter 2D
			units::RectFd					filter2d_zone_;
			units::RectFd					filter2d_subzone_;

			//! Lens used for fresnel transform (During FFT1 and FFT2)
			cuda_tools::UniquePtr<cufftComplex>		gpu_lens_;
			//! Size of a size of the lens (lens is always a square)
			uint lens_side_size_ = { 0 };
			//! Lens Queue. Used for displaying the lens.
			std::unique_ptr<Queue>				gpu_lens_queue_;
			//! Filter 2D buffer. Contains one frame.
			cuda_tools::UniquePtr<cufftComplex>	gpu_filter2d_buffer_;

			/// Vector function in which we insert the processing
			FunctionVector&					fn_compute_vect_;
			//! Main buffers
			const CoreBuffersEnv&				buffers_;
			/// Describes the frame size
			const camera::FrameDescriptor&	fd_;
			//! Compute Descriptor
			ComputeDescriptor&				cd_;
			//! Pland 2D. Used by FFTs (1, 2, filter2D).
			cuda_tools::CufftHandle&		spatial_transformation_plan_;
			//! Batch environment.
			const BatchEnv& 				batch_env_;
			//! Time transformation environment.
			TimeTransformationEnv&					time_transformation_env_;
		};
	}
}
